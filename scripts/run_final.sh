#!/usr/bin/env bash
# Recovery pass, training, and the final evaluation.
#   nohup bash scripts/run_final.sh > logs/run_final.log 2>&1 &
#
# Assumes the model is already cached and HF_HUB_OFFLINE=1 is exported.
cd "$(dirname "$0")/.."
mkdir -p logs

step () { echo; echo "### $(date '+%H:%M') — $*"; }

save () {
  step "saving"
  git add -A
  git commit -m "final run $(date '+%Y-%m-%d %H:%M')" || echo "(nothing to commit)"
  git push || echo "!!! PUSH FAILED — scp the results off before destroying"
}
trap save EXIT

step "recovering unformatted answers"
python scripts/recover_answers.py \
  --arm base --arm prompt_minimal --arm prompt_resist \
  --arm prompt_specific --arm prompt_verify || exit 1

step "training on a 50/50 mix"
python -m episteme.train --name run_50_50 --hold-firm-percent 50 --epochs 3 || exit 1

step "measuring the trained model"
python -m episteme.evaluate --name run_50_50 --adapter checkpoints/run_50_50 || exit 1

step "all arms"
python - <<'PY'
import json, pathlib
print(f"\n{'arm':<18}{'caved':>8}{'corrected':>11}{'gap':>8}{'blanks':>9}")
for d in sorted(pathlib.Path("results").iterdir()):
    f = d / "metrics.json"
    if not f.exists() or d.name in ("sanity", "base_768tok"):
        continue
    m = json.loads(f.read_text())
    v = lambda x: "   n/a" if x is None else f"{x:6.3f}"
    print(f"{d.name:<18}{v(m['caved']['value'])}{v(m['corrected']['value']):>11}"
          f"{v(m['pressure_gap']['gap']):>8}{m['missing']['overall']['value']:>8.1%}")
PY
echo
echo "adapter is in checkpoints/run_50_50 — scp it off before destroying the pod"
