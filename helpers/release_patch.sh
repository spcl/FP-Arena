#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release vX.Y.Z -> vX.Y.(Z+1) of this repository's PyPI package, from a clean main that equals origin/main and
# passed CI. It bumps [project] version in pyproject.toml (and relocks), adds a CHANGELOG.md section listing the
# commits since the last tag unless one for the new version is already there, commits, tags, builds and checks the
# wheel and sdist, pushes main and the tag, creates the GitHub release with the dists attached, and uploads to PyPI.
#   helpers/release_patch.sh            # release
#   helpers/release_patch.sh --dry-run  # print the plan and build + check the current tree; change nothing
# Needs gh (authenticated), uv, and for the upload TWINE_USER and TWINE_TOKEN in the environment.
# The same script lives in dace-relayout, dace-alloy and tactics; keep the copies identical.
set -euo pipefail
ulimit -c 0
die() { echo "release_patch.sh: $*" >&2; exit 1; }
cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
dry=0
case "${1:-}" in
    --dry-run) dry=1 ;;
    "") ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

[[ "$(git branch --show-current)" == main ]] || die "not on main"
[[ -z "$(git status --porcelain)" ]] || die "the working tree has uncommitted changes"
git fetch -q origin main --tags
head="$(git rev-parse HEAD)"
[[ "${head}" == "$(git rev-parse origin/main)" ]] || die "main is not origin/main; pull or push first"
slug="$(gh repo view --json nameWithOwner -q .nameWithOwner)"

project() { python3 -c 'import sys, tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"][sys.argv[1]])' "$1"; }
name="$(project name)"
old="$(project version)"
[[ "${old}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "version '${old}' is not X.Y.Z"
new="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
tag="v${new}"
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null && die "tag ${tag} exists already"
[[ "$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/${name}/${new}/json")" == 404 ]] ||
    die "${name} ${new} is on PyPI already"
previous="v${old}"
git rev-parse -q --verify "refs/tags/${previous}" >/dev/null || die "no tag ${previous} for the current version"
commits="$(git log --no-merges --format='- %s (%h)' "${previous}..HEAD")"
[[ -n "${commits}" ]] || { echo "${name}: no commits since ${previous}, nothing to release"; exit 0; }

# Only a commit every workflow passed on is released.
runs="$(gh run list -R "${slug}" -c "${head}" -L 100 --json workflowName,status,conclusion)"
[[ "$(jq length <<<"${runs}")" -gt 0 ]] || die "no CI run on ${head:0:9}"
open="$(jq -r '.[] | select(.status != "completed" or (.conclusion != "success" and .conclusion != "skipped"))
    | "  \(.workflowName): \(.status) \(.conclusion)"' <<<"${runs}")"
[[ -z "${open}" ]] || die "CI on ${head:0:9} is not green:"$'\n'"${open}"

echo "=== ${name} ${old} -> ${new} (${slug}, main ${head:0:9}) ==="
echo "${commits}"
work="$(mktemp -d "${TMPDIR:-/tmp}/release-patch.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
if ((dry)); then
    uv build -q -o "${work}/dist"
    uvx twine check --strict "${work}"/dist/*
    echo "dry run: nothing changed; rerun without --dry-run to release ${tag}"
    exit 0
fi
[[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] || die "TWINE_USER and TWINE_TOKEN must be set"

sed -i "0,/^version = \"${old}\"$/s//version = \"${new}\"/" pyproject.toml
[[ "$(project version)" == "${new}" ]] || die "could not set version ${new} in pyproject.toml"
[[ ! -f uv.lock ]] || uv lock -q
if ! grep -q "^## ${new} " CHANGELOG.md; then
    section="## ${new} ($(date +%F))"$'\n\n'"${commits}"$'\n'
    python3 - "${section}" <<'EOF'
import pathlib
import sys

path = pathlib.Path("CHANGELOG.md")
text = path.read_text()
at = text.find("\n## ")
at = len(text) if at < 0 else at + 1
path.write_text(text[:at] + sys.argv[1] + "\n" + text[at:])
EOF
fi
notes="$(awk -v v="## ${new} " 'index($0, v) == 1 { on = 1; next } on && /^## / { exit } on' CHANGELOG.md)"
git add -A
git commit -q -m "Release ${tag}"
git tag -a "${tag}" -m "${name} ${new}"
uv build -q -o "${work}/dist"
uvx twine check --strict "${work}"/dist/*

git push -q origin main "${tag}"
gh release create "${tag}" -R "${slug}" --title "${name} ${new}" --notes "${notes}" "${work}"/dist/*
TWINE_USERNAME="${TWINE_USER}" TWINE_PASSWORD="${TWINE_TOKEN}" uvx twine upload --non-interactive "${work}"/dist/*
echo "released ${name} ${new}: https://github.com/${slug}/releases/tag/${tag} https://pypi.org/project/${name}/${new}/"
