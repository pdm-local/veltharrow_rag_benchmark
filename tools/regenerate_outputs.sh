#!/usr/bin/env bash
# Rebuild every derived output from the canonical result CSVs, in dependency order.
# Run after any change to results/*/data/raw_results.csv (a rerun, a reparse).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

echo "== validating canonical CSVs =="
# A live run_benchmark.py appends rows while this script reads them, so every derived
# output would be built from a moving snapshot. score_unanswered.py and
# archive_stale_conditions.py already refuse under these conditions; so does this.
if pgrep -af "run_benchmark.py" >/dev/null 2>&1; then
  echo "  ERROR: run_benchmark.py is live. The canonical CSVs are being appended to;"
  echo "         regenerating now would publish tables and figures from a partial run."
  pgrep -af "run_benchmark.py" | sed "s|^|         |"
  echo "         Wait for the run to finish (and dedupe) before regenerating."
  exit 1
fi

python3 - <<'PY' || exit 1
import csv, sys, collections
csv.field_size_limit(sys.maxsize)
bad = False
for tier in ("small", "medium"):
    rows = list(csv.DictReader(open(f"results/{tier}/data/raw_results.csv")))
    keys = [(r["document_id"], r["question_id"], r["model_label"], r["retrieval_condition"])
            for r in rows]
    dupes = len(keys) - len(set(keys))
    blank = sum(1 for r in rows if not r["predicted_letter"].strip())
    cells = collections.Counter((r["model_label"], r["retrieval_condition"]) for r in rows)
    off = {c: n for c, n in cells.items() if n != 400}
    print(f"  {tier}: {len(rows)} rows | {len(cells)} cells | {dupes} duplicate keys | "
          f"{blank} blank letters | {len(off)} cells not at n=400")
    for (m, c), n in sorted(off.items(), key=lambda kv: kv[1])[:10]:
        print(f"     {m:38s} {c:28s} n={n}")
    if dupes or blank or off:
        print(f"  ERROR: {tier} has duplicate keys, unparsed rows, or off-size cells "
              f"in the canonical CSV")
        bad = True
sys.exit(1 if bad else 0)
PY

python3 tools/audit_paper_results.py

echo "== tests =="
python3 -m pytest tests/ -q || exit 1

echo "== retrieval quality and decomposition =="
python3 scripts/compute_retrieval_quality.py
for t in small medium; do
  python3 scripts/plot_retrieval_quality.py --tier "$t" --out-dir "figures/$t/retrieval" --dpi 300
done

echo "== plots (300 dpi) =="
for t in small medium; do
  python3 scripts/plot_results.py --tier "$t" --out-dir "results/$t/plots/publication" --dpi 300 \
    2>&1 | grep -E "Dropping|n=|Done"
done

echo "== syncing committed overview figures =="
for t in small medium; do
  src="results/$t/plots/publication/qwen3_0_6b"
  n=0
  for f in $(cd "figures/$t" && find . -name "*.png" -not -path "./retrieval/*"); do
    if [ -f "$src/$f" ]; then cp "$src/$f" "figures/$t/$f"; n=$((n+1));
    else echo "  MISSING in plots: $t/$f"; fi
  done
  echo "  $t: synced $n figures"
done

echo "== main + free-form figures (figures/main, figures/freeform) =="
# These read results/ directly and write straight into the committed figures/ tree, so
# unlike the tier plots above they need no sync step. They live here because omitting
# them once left figures/main/recognition_vs_production.png a refill behind the tables.
python3 tools/plot_recognition_gap.py 2>&1 | sed "s|^|  |"
python3 tools/plot_quality_latency.py 2>&1 | sed "s|^|  |"
python3 scripts/plot_freeform.py --out-dir figures/freeform --dpi 300 2>&1 | tail -3 | sed "s|^|  |"

echo "== paper tables =="
python3 tools/paper_tables.py > results/rerun_logs/paper_tables.txt 2>&1 \
  && echo "  -> results/rerun_logs/paper_tables.txt"
grep -A20 "WARNING - cells with fewer" results/rerun_logs/paper_tables.txt | head -30 || true

# main.tex \input's these, so the manuscript's tables follow the data automatically.
if [ -d paper/article ]; then
  python3 tools/paper_tables.py --latex | sed "s|^|  |"
  ( cd paper/article && latexmk -pdf -interaction=nonstopmode -bibtex main.tex >/dev/null 2>&1 \
      && echo "  rebuilt paper/article/main.pdf" ) \
    || { echo "  ERROR: paper build failed"; exit 1; }
  python3 paper/article/export_google_docs.py
fi

echo "== statistical tests =="
python3 tools/stats_tests.py | tee results/rerun_logs/stats_tests.txt

echo "== done; review paper_tables.txt and stats_tests.txt before editing main.tex =="
