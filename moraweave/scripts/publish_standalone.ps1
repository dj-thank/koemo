[CmdletBinding()]
param(
    [string]$Owner = "dj-thank",
    [string]$Repository = "moraweave",
    [string]$SourceRepository = "https://github.com/dj-thank/koemo.git",
    [string]$SourceBranch = "public/moraweave-v0.1.0"
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is required" }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "GitHub CLI is required" }
& gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) { throw "GitHub CLI is not authenticated" }

& gh repo view "$Owner/$Repository" *> $null
if ($LASTEXITCODE -eq 0) {
    throw "Refusing to overwrite existing repository: $Owner/$Repository"
}

$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("moraweave-publish-" + [guid]::NewGuid())
try {
    & git clone --branch $SourceBranch --single-branch $SourceRepository $Work
    if ($LASTEXITCODE -ne 0) { throw "clone failed" }
    Push-Location $Work
    try {
        & git subtree split --prefix=moraweave -b standalone-main
        if ($LASTEXITCODE -ne 0) { throw "subtree split failed" }
        & gh repo create "$Owner/$Repository" --public --description "Mora-aware evidence-fused Japanese speech transcription with selective re-listening"
        if ($LASTEXITCODE -ne 0) { throw "repository creation failed" }
        & git push "https://github.com/$Owner/$Repository.git" "standalone-main:main"
        if ($LASTEXITCODE -ne 0) { throw "push failed" }
        Write-Host "Published: https://github.com/$Owner/$Repository"
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Work
}
