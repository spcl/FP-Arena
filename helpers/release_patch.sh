#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release vX.Y.Z -> vX.Y.(Z+1) of this repository's PyPI package, from a clean main that equals origin/main and
# passed CI. It bumps [project] version in pyproject.toml (and relocks), adds a CHANGELOG.md section listing the
# commits since the last tag unless one for the new version is already there, commits, tags, builds and checks the
# wheel and sdist, pushes main and the tag, uploads the dists to PyPI, and then creates the GitHub release with the same
# dists attached. Both uploads run here, from one build; no workflow uploads anything.
#   helpers/release_patch.sh            # release
#   helpers/release_patch.sh --dry-run  # print the plan and build + check the current tree; change nothing
#   helpers/release_patch.sh --github-only  # everything but the PyPI upload: the GitHub release carries the dists (no
#                                       # TWINE_* needed); the next run without it first uploads exactly those to PyPI
# Needs gh (authenticated), uv, and for the upload TWINE_USER and TWINE_TOKEN in the environment (never in a file).
# The same script lives in dace-relayout, dace-alloy, dace-fortran, tactics and fp-arena; keep the copies identical.
set -euo pipefail
ulimit -c 0
die() { echo "release_patch.sh: $*" >&2; exit 1; }
cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
dry=0
github_only=0
case "${1:-}" in
    --dry-run) dry=1 ;;
    --github-only) github_only=1 ;;
    "") ;;
    *) echo "usage: $0 [--dry-run | --github-only]" >&2; exit 2 ;;
esac

[[ "$(git branch --show-current)" == main ]] || die "not on main"
[[ -z "$(git status --porcelain)" ]] || die "the working tree has uncommitted changes"
git fetch -q origin main --tags
head="$(git rev-parse HEAD)"
[[ "${head}" == "$(git rev-parse origin/main)" ]] || die "main is not origin/main; pull or push first"
slug="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
pins_pr="$(gh pr list -R "${slug}" --state open --json number,headRefName,url -q '.[] | select(.headRefName | startswith("pins/")) | .url')"
[[ -z "${pins_pr}" ]] || die "a pin-bump PR is open (${pins_pr}): merge it once its CI is green, then rerun"

# Phase 1: our packages this repository depends on move to their newest releases first, so a release never ships stale
# pins: each `>=` lower bound rises to the newest PyPI version, and a [tool.uv.sources] git pin of one of them is
# dropped, so the package resolves to that PyPI release: our packages pin released versions, never git tags or commits.
# A pin bump is a change like any other, tested where it lands: when anything moved, the bump goes to a pins/*
# branch and a pull request to main, and the script stops. Merge the PR once its CI is green, then rerun; when that CI
# fails because a dependency's release broke this repository, fix it in the dependency and release that again. Release order (dependencies first): dace-extended, dace-fortran, fp-arena,
# dace-relayout and dace-alloy, hpcagent-bench, tactics.
bumps="$(python3 - "${dry}" <<'EOF'
import json
import pathlib
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request

OURS = ["dace-extended", "dace-fortran", "fp-arena", "dace-relayout", "dace-alloy", "tactics", "hpcagent-bench"]
dry = sys.argv[1] == "1"
path = pathlib.Path("pyproject.toml")
text = path.read_text()
own = tomllib.loads(text)["project"]["name"]


def newest(name: str) -> str:
    """The newest PyPI version of ``name``, or "" while it has none (the query string skips the CDN's stale copy)."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json?fresh={time.time_ns()}") as reply:
            return json.load(reply)["info"]["version"]
    except urllib.error.HTTPError:
        return ""


for name in OURS:
    if name == own:
        continue
    bound = re.compile(rf'(["\']{re.escape(name)}(?:\[[^\]]*\])?>=)([0-9][0-9.]*)')
    source = re.compile(rf'^{re.escape(name)} = \{{ git = "([^"]+)", (?:rev|tag|branch) = "([^"]+)" \}}\n', re.MULTILINE)
    if not bound.search(text) and not source.search(text):
        continue
    version = newest(name)
    if not version:
        continue
    for match in bound.finditer(text):
        if match.group(2) != version:
            print(f"{name}>={match.group(2)} -> >={version}")
    text = bound.sub(rf"\g<1>{version}", text)
    match = source.search(text)
    if match is not None:
        print(f"{name} source git {match.group(2)[:12]} -> PyPI {version}")
        text = text[: match.start()] + text[match.end() :]
# A sources table left empty goes too, with the comment lines right above it
text = re.sub(r"(?:^#[^\n]*\n)*^\[tool\.uv\.sources\]\n(?=\n|\[|\Z)", "", text, flags=re.MULTILINE)
if not dry:
    path.write_text(text)
EOF
)"
if [[ -n "${bumps}" ]]; then
    echo "=== our packages' newest releases ==="
    echo "${bumps}"
    if ((dry)); then
        echo "dry run: a release first opens a pull request with these bumps and stops for its CI"
        exit 0
    fi
    [[ ! -f uv.lock ]] || uv lock -q --refresh
    branch="pins/$(date +%Y%m%d-%H%M%S)"
    git switch -q -c "${branch}"
    git add -A
    git commit -q -m "Pin the newest releases of our packages" -m "${bumps}"
    git push -q origin "${branch}"
    pr="$(gh pr create -R "${slug}" --base main --head "${branch}" --title "Pin the newest releases of our packages" \
        --body "${bumps}")"
    git switch -q main
    echo "opened ${pr}; merge it once its CI is green, then rerun $0"
    exit 0
fi

((dry || github_only)) || [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] ||
    die "TWINE_USER and TWINE_TOKEN must be set (or pass --github-only)"
work="$(mktemp -d "${TMPDIR:-/tmp}/release-patch.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
project() { python3 -c 'import sys, tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"][sys.argv[1]])' "$1"; }
on_pypi() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/${name}/$1/json")" != 404 ]]; }
upload() { TWINE_USERNAME="${TWINE_USER}" TWINE_PASSWORD="${TWINE_TOKEN}" uvx twine upload --non-interactive "$@"; }
name="$(project name)"
old="$(project version)"
[[ "${old}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "version '${old}' is not X.Y.Z"
new="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
tag="v${new}"
# A release made with --github-only reaches PyPI on the next run without it: exactly the dists its GitHub release
# carries, before anything new is released.
if ! on_pypi "${old}" && gh release view "v${old}" -R "${slug}" >/dev/null 2>&1; then
    if ((dry || github_only)); then
        echo "v${old} is on GitHub only; a run without --github-only uploads its dists to PyPI first"
    else
        gh release download "v${old}" -R "${slug}" -D "${work}/pending" -p '*.whl' -p '*.tar.gz'
        upload --skip-existing "${work}"/pending/*
        echo "uploaded the GitHub-only v${old}: https://pypi.org/project/${name}/${old}/"
    fi
fi
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null && die "tag ${tag} exists already"
! on_pypi "${new}" || die "${name} ${new} is on PyPI already"
git rev-parse -q --verify "refs/tags/v${old}" >/dev/null || die "no tag v${old} for the current version"
commits="$(git log --no-merges --format='- %s (%h)' "v${old}..HEAD")"
[[ -n "${commits}" ]] || { echo "${name}: no commits since v${old}, nothing to release"; exit 0; }

# Only a commit every workflow passed on is released.
runs="$(gh run list -R "${slug}" -c "${head}" -L 100 --json workflowName,status,conclusion)"
[[ "$(jq length <<<"${runs}")" -gt 0 ]] || die "no CI run on ${head:0:9}"
open="$(jq -r '.[] | select(.status != "completed" or (.conclusion != "success" and .conclusion != "skipped"))
    | "  \(.workflowName): \(.status) \(.conclusion)"' <<<"${runs}")"
[[ -z "${open}" ]] || die "CI on ${head:0:9} is not green:"$'\n'"${open}"

echo "=== ${name} ${old} -> ${new} (${slug}, main ${head:0:9}) ==="
echo "${commits}"
if ((dry)); then
    uv build -q -o "${work}/dist"
    uvx twine check --strict "${work}"/dist/*
    echo "dry run: nothing changed; rerun without --dry-run to release ${tag}"
    exit 0
fi
sed -i "0,/^version = \"${old}\"$/s//version = \"${new}\"/" pyproject.toml
[[ "$(project version)" == "${new}" ]] || die "could not set version ${new} in pyproject.toml"
[[ ! -f uv.lock ]] || uv lock -q
if ! grep -q "^## ${new} " CHANGELOG.md; then
    # The new section goes above the newest one, or at the end.
    SECTION="## ${new} ($(date +%F))"$'\n\n'"${commits}"$'\n' awk '
        !done && /^## / { print ENVIRON["SECTION"]; done = 1 } { print } END { if (!done) print ENVIRON["SECTION"] }
    ' CHANGELOG.md >"${work}/CHANGELOG.md"
    mv "${work}/CHANGELOG.md" CHANGELOG.md
fi
notes="$(awk -v v="## ${new} " 'index($0, v) == 1 { on = 1; next } on && /^## / { exit } on' CHANGELOG.md)"
git add -A
git commit -q -m "Release ${tag}"
git tag -a "${tag}" -m "${name} ${new}"
uv build -q -o "${work}/dist"
uvx twine check --strict "${work}"/dist/*

git push -q origin main "${tag}"
((github_only)) || upload "${work}"/dist/*
gh release create "${tag}" -R "${slug}" --verify-tag --title "${name} ${new}" --notes "${notes}" --generate-notes \
    "${work}"/dist/*
echo "released ${name} ${new}: https://github.com/${slug}/releases/tag/${tag}"
if ((github_only)); then echo "GitHub only: rerun $0 to upload it to PyPI"; else echo "https://pypi.org/project/${name}/${new}/"; fi
