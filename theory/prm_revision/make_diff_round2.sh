#!/usr/bin/env bash
# Regenerate the round 2 marked-up diffs against the frozen round 1 sources.
# Usage: ./make_diff_round2.sh [-n]   (-n prints the commands without running)
set -euo pipefail
cd "$(dirname "$0")"

DRY=0
[ "${1:-}" = "-n" ] && DRY=1

run() {
  echo "+ $*"
  [ "$DRY" -eq 1 ] || eval "$@"
}

run "latexdiff --math-markup=whole --append-textcmd=runinsec round1/prm_main.tex prm_main.tex > prm_main_diff_round2.tex"
run "latexdiff --math-markup=whole round1/prm_supplemental.tex prm_supplemental.tex > prm_supplemental_diff_round2.tex"
run "latexmk -pdf -interaction=nonstopmode -silent prm_main_diff_round2.tex"
run "latexmk -pdf -interaction=nonstopmode -silent prm_supplemental_diff_round2.tex"
