#!/usr/bin/env bash
set -euo pipefail

OWNER="${1:-dj-thank}"
REPO="${2:-moraweave}"
SOURCE_BRANCH="${SOURCE_BRANCH:-public/moraweave-v0.1.0}"
SOURCE_REPO="${SOURCE_REPO:-https://github.com/dj-thank/koemo.git}"

command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v gh >/dev/null || { echo "GitHub CLI is required" >&2; exit 2; }
gh auth status >/dev/null

if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing repository: $OWNER/$REPO" >&2
  exit 3
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --branch "$SOURCE_BRANCH" --single-branch "$SOURCE_REPO" "$WORK/source"
cd "$WORK/source"
git subtree split --prefix=moraweave -b standalone-main

gh repo create "$OWNER/$REPO" \
  --public \
  --description "Mora-aware evidence-fused Japanese speech transcription with selective re-listening"

git push "https://github.com/$OWNER/$REPO.git" standalone-main:main

echo "Published: https://github.com/$OWNER/$REPO"
