#!/usr/bin/env bash
# Everything that needs a GPU tonight. Run it and leave it.
#   nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
#
# Whatever happens -- finished, failed, or interrupted -- the results that exist
# get committed and pushed on the way out, so nothing is lost when the instance
# is destroyed.
cd "$(dirname "$0")/.."
mkdir -p logs

step () { echo; echo "### $(date '+%H:%M') — $*"; }

save_and_push () {
  step "saving whatever we have"
  git add -A
  git commit -m "cloud run $(date '+%Y-%m-%d %H:%M')" || echo "(nothing new to commit)"
  git push || echo "!!! PUSH FAILED — results are committed locally, scp them off"
  echo
  echo "results on disk:"
  ls results/ 2>/dev/null
  du -sh data/train checkpoints 2>/dev/null
}
trap save_and_push EXIT

summary () {
  python - <<'PY'
import json, pathlib
rows = []
for d in sorted(pathlib.Path("results").iterdir()):
    f = d / "metrics.json"
    if f.exists():
        m = json.loads(f.read_text())
        rows.append((d.name, m["caved"]["value"], m["corrected"]["value"],
                     m["pressure_gap"]["gap"], m["missing"]["overall"]["value"]))
print(f"\n{'arm':<18}{'caved':>8}{'corrected':>11}{'gap':>8}{'blanks':>9}")
for name, c, k, g, b in rows:
    f = lambda v: "  n/a" if v is None else f"{v:6.3f}"
    print(f"{name:<18}{f(c)}{f(k):>11}{f(g):>8}{b:>8.1%}")
PY
}

# the pool has to exist before anything can be evaluated on it
step "building the item pool"
python -m episteme.data || exit 1

step "sanity check, 40 items"
python -m episteme.evaluate --name sanity --limit 40 --prompt base || exit 1

step "base arm"
python -m episteme.evaluate --name base --prompt base || exit 1

step "training examples — the long one, ~3.5h"
python -m episteme.pairs --samples 4 || exit 1
python -m episteme.pairs --rebuild --pairs-per-item 2 || exit 1

# arms are independent: one failing must not cost the others
for arm in minimal resist specific verify; do
  step "$arm arm"
  python -m episteme.evaluate --name "prompt_$arm" --prompt "$arm" \
    || echo "!!! $arm failed, carrying on"
done

step "all done"
summary
echo
echo "tomorrow:"
echo "  python -m episteme.train --name run_50_50 --hold-firm-percent 50 --epochs 3"
echo "  python -m episteme.evaluate --name run_50_50 --adapter checkpoints/run_50_50"
