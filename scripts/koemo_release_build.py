"""Koemo public release build/sign/package helper.

既定は署名必須。署名基盤がない内部確認だけ `--unsigned-beta` を明示する。
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist" / "Koemo"
EXE = DIST_DIR / "Koemo.exe"
RELEASE_DIR = ROOT / "release"
INSTALLER_SCRIPT = ROOT / "packaging" / "koemo.iss"
RELEASE_NOTES_TEMPLATE = ROOT / "packaging" / "release-notes-ja.md"
VERSION = "0.1.0-rc1"
APP_NAME = "Koemo"
TIMESTAMP_URL = "http://timestamp.acs.microsoft.com"


def run(cmd, *, check=True, env=None):
    print("+ " + " ".join(str(c) for c in cmd))
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def tool(name, env_name=None):
    if env_name and os.environ.get(env_name):
        path = Path(os.environ[env_name])
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if found:
        return found
    return None


def ensure_inno(args):
    iscc = tool("iscc", "KOEMO_ISCC")
    if iscc:
        return iscc
    if not args.install_tools:
        raise SystemExit("ISCC.exe が見つかりません。Inno Setupを入れるか KOEMO_ISCC を指定してください。")
    winget = tool("winget")
    if not winget:
        raise SystemExit("winget が見つからないため Inno Setup を自動導入できません。")
    run([winget, "install", "--id", "JRSoftware.InnoSetup", "-e", "-s", "winget", "--accept-package-agreements", "--accept-source-agreements"])
    iscc = tool("iscc", "KOEMO_ISCC")
    if not iscc:
        raise SystemExit("Inno Setup導入後も ISCC.exe が見つかりません。KOEMO_ISCC を指定してください。")
    return iscc


def ensure_signing(args):
    if args.unsigned_beta:
        return None
    signtool = tool("signtool", "KOEMO_SIGNTOOL")
    dlib = os.environ.get("KOEMO_SIGNING_DLIB", "")
    metadata = os.environ.get("KOEMO_SIGNING_METADATA", "")
    if not signtool:
        raise SystemExit("signtool.exe が見つかりません。KOEMO_SIGNTOOL を指定してください。")
    if not dlib or not Path(dlib).is_file():
        raise SystemExit("KOEMO_SIGNING_DLIB に Azure Artifact Signing dlib DLL を指定してください。")
    if not metadata or not Path(metadata).is_file():
        raise SystemExit("KOEMO_SIGNING_METADATA に repo外の Azure signing metadata JSON を指定してください。")
    metadata_path = Path(metadata).resolve()
    try:
        metadata_path.relative_to(ROOT.resolve())
        raise SystemExit("KOEMO_SIGNING_METADATA は repo 内に置かないでください。")
    except ValueError:
        pass
    return {"signtool": signtool, "dlib": dlib, "metadata": metadata}


def build_pyinstaller(skip_build=False):
    if skip_build:
        if not EXE.is_file():
            raise SystemExit("dist\\Koemo\\Koemo.exe がありません。--skip-build は使えません。")
        return
    run([sys.executable, "-m", "PyInstaller", "koemo.spec", "--noconfirm", "--clean"])
    if not EXE.is_file():
        raise SystemExit("PyInstaller後に dist\\Koemo\\Koemo.exe が見つかりません。")


def collect_pe_files():
    suffixes = {".exe", ".dll", ".pyd"}
    return sorted(p for p in DIST_DIR.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def sign_file(path, signing):
    cmd = [
        signing["signtool"], "sign", "/v", "/fd", "SHA256",
        "/tr", TIMESTAMP_URL, "/td", "SHA256",
        "/dlib", signing["dlib"], "/dmdf", signing["metadata"],
        "/d", APP_NAME, str(path),
    ]
    run(cmd)


def verify_signature(path, signing):
    run([signing["signtool"], "verify", "/pa", "/all", str(path)])
    literal = str(path).replace("'", "''")
    ps = [
        "powershell", "-NoProfile", "-Command",
        f"$s=Get-AuthenticodeSignature -LiteralPath '{literal}'; "
        "if ($s.Status -ne 'Valid') { Write-Error $s.StatusMessage; exit 1 }",
    ]
    run(ps)


def sign_and_verify_dist(signing):
    files = collect_pe_files()
    if not files:
        raise SystemExit("署名対象のPEファイルが見つかりません。")
    for path in files:
        sign_file(path, signing)
    for path in files:
        verify_signature(path, signing)
    return files


def make_zip(signed):
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "signed" if signed else "UNSIGNED-BETA"
    zip_path = RELEASE_DIR / f"Koemo-{VERSION}-{suffix}-portable.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(DIST_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(DIST_DIR.parent))
    return zip_path


def build_installer(iscc, signed):
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    if not INSTALLER_SCRIPT.is_file():
        raise SystemExit(f"installer script missing: {INSTALLER_SCRIPT}")
    suffix = "" if signed else "-UNSIGNED-BETA"
    run([
        iscc,
        "/Qp",
        f"/DMyAppVersion={VERSION}",
        f"/DMyOutputSuffix={suffix}",
        str(INSTALLER_SCRIPT),
    ])
    installer = RELEASE_DIR / f"Koemo-{VERSION}{suffix}-Setup.exe"
    if not installer.is_file():
        raise SystemExit(f"installer not created: {installer}")
    return installer


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_release_metadata(artifacts, signed, signed_files):
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    notes = RELEASE_NOTES_TEMPLATE.read_text(encoding="utf-8")
    notes = notes.replace("{{VERSION}}", VERSION)
    notes = notes.replace("{{SIGNED_STATUS}}", "署名済み" if signed else "未署名内部ベータ")
    notes_path = RELEASE_DIR / f"Koemo-{VERSION}-RELEASE-NOTES-ja.md"
    notes_path.write_text(notes, encoding="utf-8")
    all_artifacts = list(artifacts) + [notes_path]
    sums_path = RELEASE_DIR / "SHA256SUMS.txt"
    sums_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in all_artifacts),
        encoding="utf-8",
    )
    metadata = {
        "app": APP_NAME,
        "version": VERSION,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "signed": signed,
        "signing": "Azure Artifact Signing" if signed else "UNSIGNED-BETA",
        "artifacts": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in all_artifacts],
        "signed_pe_count": len(signed_files),
        "smart_screen_note": "署名済み非Store配布でもSmartScreen reputationが育つまで警告が出る場合があります。",
    }
    (RELEASE_DIR / f"Koemo-{VERSION}-release.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sums_path


def secret_scan(paths):
    suspicious = []
    patterns = (
        "s" + "k-", "gh" + "p_", "github_" + "pat_", "xox" + "b-",
        "AK" + "IA", "-----BEGIN " + "PRIVATE KEY-----",
    )
    for base in paths:
        if not Path(base).exists():
            continue
        files = [Path(base)] if Path(base).is_file() else [p for p in Path(base).rglob("*") if p.is_file()]
        for path in files:
            if "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in {".exe", ".dll", ".pyd", ".pyc", ".wav", ".zip", ".ico", ".png", ".pdf", ".docx"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if any(p in text for p in patterns):
                suspicious.append(str(path))
    if suspicious:
        raise SystemExit("secret scan failed:\n" + "\n".join(sorted(set(suspicious))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsigned-beta", action="store_true", help="署名なし内部ベータとしてartifact名に明示する")
    parser.add_argument("--skip-build", action="store_true", help="既存 dist\\Koemo を使う")
    parser.add_argument("--install-tools", action="store_true", help="wingetでInno Setupを導入する")
    parser.add_argument("--skip-installer", action="store_true", help="portable zipのみ作る")
    args = parser.parse_args()

    signing = ensure_signing(args)
    build_pyinstaller(skip_build=args.skip_build)
    signed_files = sign_and_verify_dist(signing) if signing else []
    zip_path = make_zip(signed=bool(signing))
    artifacts = [zip_path]
    if not args.skip_installer:
        iscc = ensure_inno(args)
        installer = build_installer(iscc, signed=bool(signing))
        if signing:
            sign_file(installer, signing)
            verify_signature(installer, signing)
        artifacts.append(installer)
    secret_scan([ROOT / "koemo", ROOT / "scripts", ROOT / "packaging", RELEASE_DIR])
    sums = write_release_metadata(artifacts, signed=bool(signing), signed_files=signed_files)
    print(f"[OK] release artifacts: {RELEASE_DIR}")
    print(f"[OK] checksums: {sums}")
    if not signing:
        print("[WARN] UNSIGNED-BETA: 世界配布readyとは扱わないでください。")


if __name__ == "__main__":
    main()
