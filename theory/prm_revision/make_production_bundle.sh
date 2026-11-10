#!/usr/bin/env bash
# Build the APS production upload bundle (unmarked TeX source, one file per
# figure, frozen .bbl, Supplemental Material PDF) in production/.
# Usage: ./make_production_bundle.sh [-n]   (-n prints the commands without running)
set -euo pipefail
cd "$(dirname "$0")"

DRY=0
[ "${1:-}" = "-n" ] && DRY=1

run() {
  echo "+ $*"
  [ "$DRY" -eq 1 ] || eval "$@"
}

OUT=production
FIG1=prm_oxygen_beta_vs_charge_box_cn4_cn6.png
FIG2=thomas_fermi_reff_prl.png

run "latexmk -pdf -interaction=nonstopmode -silent prm_main.tex"
run "latexmk -pdf -interaction=nonstopmode -silent prm_supplemental.tex"

run "rm -rf $OUT && mkdir $OUT"
# Flat layout: figures renamed by figure number, no graphicspath, no
# internal revision comment on line 1.
run "sed -e '1{/^% Second revision/d;}' -e '/^\\\\graphicspath/d' -e 's/{$FIG1}/{fig1.png}/' -e 's/{$FIG2}/{fig2.png}/' prm_main.tex > $OUT/prm_main.tex"
run "cp figures/$FIG1 $OUT/fig1.png"
run "cp figures/$FIG2 $OUT/fig2.png"
run "cp prm_main.bbl $OUT/prm_main.bbl"
run "cp prm_supplemental.pdf $OUT/supplemental_material.pdf"

[ "$DRY" -eq 1 ] && { echo "+ (validate bundle in a clean temp dir, then tar)"; exit 0; }

# Checks on the bundled source.
grep -q 'graphicspath\|DIFadd\|DIFdel\|\\todo' $OUT/prm_main.tex && { echo "FAIL: markup or graphicspath left in bundle source"; exit 1; }
[ "$(grep -c 'includegraphics' $OUT/prm_main.tex)" -eq 2 ] || { echo "FAIL: expected 2 figures"; exit 1; }
grep -q 'includegraphics.*{fig1.png}' $OUT/prm_main.tex && grep -q 'includegraphics.*{fig2.png}' $OUT/prm_main.tex || { echo "FAIL: figure rename"; exit 1; }
grep -q 'See Supplemental Material at' $OUT/prm_main.bbl || { echo "FAIL: no Supplemental Material reference in .bbl"; exit 1; }
grep -q 'cite{suppmat}' $OUT/prm_main.tex || { echo "FAIL: Supplemental Material reference not cited"; exit 1; }

# Every reference cited in the SM must be in the main reference list.
keys() { grep -o '\\cite[a-z]*{[^}]*}' "$1" | sed 's/.*{//;s/}//' | tr ',' '\n' | tr -d ' ' | sort -u; }
MISSING=$(comm -13 <(keys prm_main.tex) <(keys prm_supplemental.tex))
[ -z "$MISSING" ] || { echo "FAIL: cited only in the SM: $MISSING"; exit 1; }

# Compile the bundle alone (no .bib, no parent directory) and compare text.
TMP=$(mktemp -d)
cp $OUT/prm_main.tex $OUT/prm_main.bbl $OUT/fig1.png $OUT/fig2.png "$TMP"/
( cd "$TMP" && for i in 1 2 3; do pdflatex -interaction=nonstopmode prm_main.tex >/dev/null; done )
grep -q 'undefined' "$TMP/prm_main.log" && { echo "FAIL: undefined references in bundle build"; exit 1; }
diff <(pdftotext prm_main.pdf -) <(pdftotext "$TMP/prm_main.pdf" -) >/dev/null || { echo "FAIL: bundle PDF text differs from prm_main.pdf"; exit 1; }
rm -rf "$TMP"

tar -czf $OUT/prm_production.tar.gz -C $OUT prm_main.tex prm_main.bbl fig1.png fig2.png supplemental_material.pdf
echo "OK: $OUT/ validated ($(pdfinfo prm_main.pdf | awk '/Pages/{print $2}') pp main, $(pdfinfo prm_supplemental.pdf | awk '/Pages/{print $2}') pp SM)"
ls -l $OUT
