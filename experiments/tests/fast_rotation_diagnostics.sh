#!/usr/bin/env bash
# Launches fast_rotation_diagnostics.jl in the background on this machine's GPU, stopped after 23.5 h, logging to logs/fast_rotation_diagnostics.log.
# Tests run in priority order, so the limit cuts the least important. Rerunning skips finished tests, resumes cut-off ones and appends to the log.
#   experiments/tests/fast_rotation_diagnostics.sh [test name filters]
set -euo pipefail

cd "$(dirname "$0")/../.."
script=experiments/tests/fast_rotation_diagnostics.jl

nvidia-smi > /dev/null 2>&1 || { echo "nvidia-smi can't reach the GPU driver"; exit 1; }
if pgrep -f "julia.*fast_rotation_diagnostics.jl" > /dev/null; then
    echo "$script is already running:"; pgrep -af "julia.*fast_rotation_diagnostics.jl"; exit 1
fi
output="${EXO_OCEAN_OUTPUT:-output}"
for P in 3 5 10; do
    ls "$output"/checkpoints/res1km_P${P}_dT30_spinup_iteration*.jld2 > /dev/null 2>&1 ||
        echo "Warning: no res1km_P${P}_dT30_spinup checkpoint in $output/checkpoints; its tests will be skipped"
done
echo "Free disk space for the output (the recorded fields can reach ~20 GB):"; df -h "$output/" | tail -1

mkdir -p logs
nohup timeout 23.5h julia --project=. -t 4 "$script" "$@" >> logs/fast_rotation_diagnostics.log 2>&1 &
echo "Started (PID $!), log logs/fast_rotation_diagnostics.log"
