#!/usr/bin/env bash
# OpenEvolve pilot campaign (docs/superpowers/specs/2026-10-29-openevolve-pilot-design.md).
#   bash scripts/run_openevolve_pilot.sh check    -> GPU free, Ollama, Docker image, agy
#   bash scripts/run_openevolve_pilot.sh run      -> 18 runs (L and B first, then LC); resumable
#   bash scripts/run_openevolve_pilot.sh recheck  -> independent re-check + summary + figures
# Exit code 3: Argon quota paused the campaign; re-run `run` after the agy refresh.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
ROOT="${OE_PILOT_ROOT:-benchmarks/openevolve_pilot}"
RUNNER="${OE_PILOT_RUNNER:-.venv/bin/python -m experiments.openevolve.run_pilot}"
mkdir -p "$ROOT"
LOG="$ROOT/campaign.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

status_of() { [ -f "$1/manifest.json" ] && .venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "$1/manifest.json" 2>/dev/null; }

plan() {
  for arm in L B; do for task in circle_packing knn; do for seed in 101 202 303; do echo "$task $arm $seed"; done; done; done
  for task in circle_packing knn; do for seed in 101 202 303; do echo "$task LC $seed"; done; done
}

case "${1:-}" in
  check)
    ok=0
    others=$(ollama ps 2>/dev/null | awk 'NR>1 && $1!="gemma4:12b"{print $1}')
    [ -z "$others" ] && echo "PASS ollama: no other model loaded" || { echo "FAIL ollama: loaded $others"; ok=1; }
    gpu=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -v -i ollama)
    [ -z "$gpu" ] && echo "PASS gpu: no other compute process" || { echo "FAIL gpu: $gpu"; ok=1; }
    docker image inspect antigravity-orion:local >/dev/null 2>&1 && echo "PASS docker image" || { echo "FAIL docker image antigravity-orion:local"; ok=1; }
    command -v agy >/dev/null 2>&1 || [ -x "$HOME/.local/bin/agy" ] && echo "PASS agy" || { echo "FAIL agy not found"; ok=1; }
    exit $ok
    ;;
  run)
    [ -n "${OE_PILOT_SKIP_TIMERS:-}" ] || { systemctl --user stop archive-index-sessions.timer archive-index.timer 2>/dev/null && log "paused archive-index timers"; }
    rc=0
    while read -r task arm seed; do
      out="$ROOT/${task}_${arm}_s${seed}"
      st=$(status_of "$out")
      [ "$st" = "complete" ] && continue
      resume=(); [ -d "$out" ] && resume=(--resume)
      if [ -n "${OE_PILOT_DRY_RUN:-}" ]; then echo "PLAN --task $task --arm $arm --seed $seed --out $out ${resume[*]}"; continue; fi
      log "start $task $arm s$seed ${resume[*]} at $(git log --oneline -1)"
      $RUNNER --task "$task" --arm "$arm" --seed "$seed" --out "$out" "${resume[@]}" >> "$LOG" 2>&1
      code=$?
      if [ $code -eq 3 ]; then log "Argon quota paused the campaign at $task $arm s$seed; re-run after the agy refresh"; rc=3; break; fi
      [ $code -ne 0 ] && log "run $task $arm s$seed exited $code" && rc=$code
      log "end $task $arm s$seed status=$(status_of "$out")"
    done < <(plan)
    [ -n "${OE_PILOT_SKIP_TIMERS:-}" ] || { systemctl --user start archive-index-sessions.timer archive-index.timer 2>/dev/null && log "restored archive-index timers"; }
    exit $rc
    ;;
  recheck)
    .venv/bin/python -m experiments.openevolve.analyze "$ROOT" 2>&1 | tee -a "$LOG"
    ;;
  *) echo "usage: $0 check|run|recheck" >&2; exit 2 ;;
esac
