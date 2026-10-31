#!/usr/bin/env bash
# AutoDL: finish training and its evaluations, preserve logs, then power off.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

if [[ "${1:-}" == "--check" ]]; then
  shift
  printf 'Command: '
  printf '%q ' python -u scripts/train_expanded_smoke.py "$@"
  printf '\nNormal completion or training failure: save logs and call /usr/bin/shutdown.\n'
  printf 'Ctrl+C or SIGTERM: stop without automatic shutdown.\n'
  exit 0
fi

# Configuration/help checks must never turn off an instance.
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" || "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec python scripts/train_expanded_smoke.py "$@"
  fi
done

if [[ "$project_root" != /root/autodl-tmp/* || ! -x /usr/bin/shutdown ]]; then
  printf 'Run this script from the project under /root/autodl-tmp/ on your AutoDL instance.\n' >&2
  exit 2
fi

mkdir -p runs/cloud_logs
log_file="runs/cloud_logs/train_$(date -u +%Y%m%d_%H%M%S)_$$.log"
trap 'printf "Interrupted: automatic shutdown skipped.\n"; exit 130' INT
trap 'printf "Terminated: automatic shutdown skipped.\n"; exit 143' TERM

printf 'Log: %s\n' "$log_file"
printf 'The instance will shut down after training/evaluation finishes or fails.\n'
set +e
python -u scripts/train_expanded_smoke.py "$@" 2>&1 | tee "$log_file"
training_status=${PIPESTATUS[0]}
set -e
printf '\nTraining exit code: %s. Shutting down AutoDL; result files are retained.\n' "$training_status" | tee -a "$log_file"
sync
if ! /usr/bin/shutdown; then
  printf 'Automatic shutdown failed. Shut down the instance in the AutoDL console.\n' | tee -a "$log_file"
  exit 1
fi
exit "$training_status"
