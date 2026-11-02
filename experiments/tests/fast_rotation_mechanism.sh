#!/usr/bin/env bash
# Launches fast_rotation_mechanism.jl in the background on this machine's GPU, stopped after 17.75 h, logging to logs/fast_rotation_mechanism.log.
# Tests run in priority order, so the limit cuts the least important. Rerunning skips finished tests, resumes cut-off ones and appends to the log.
#   experiments/tests/fast_rotation_mechanism.sh [test name filters]
set -euo pipefail

cd "$(dirname "$0")/../.."
script=experiments/tests/fast_rotation_mechanism.jl

nvidia-smi > /dev/null 2>&1 || { echo "nvidia-smi can't reach the GPU driver"; exit 1; }
if pgrep -f "julia.*fast_rotation_(mechanism|diagnostics).jl" > /dev/null; then
    echo "Already running:"; pgrep -af "julia.*fast_rotation_(mechanism|diagnostics).jl"; exit 1
fi
output="${EXO_OCEAN_OUTPUT:-output}"
for run in res1km_P3_dT30_spinup res1km_P5_dT30_spinup res1km_P10_dT30_spinup fastdiag_heat_P3_cap0.5; do
    ls "$output"/checkpoints/${run}_iteration*.jld2 > /dev/null 2>&1 ||
        echo "Warning: no $run checkpoint in $output/checkpoints; tests starting from it will be skipped"
done
emulator_python="${EXO_EMULATOR_PYTHON:-/data/pt426/ocean-venv/bin/python}"
[ -x "$emulator_python" ] || echo "Warning: no emulator Python at $emulator_python (set EXO_EMULATOR_PYTHON); the wind tests will fail unless their wind file is cached"
echo "Free disk space for the output (the recorded fields can reach ~10 GB):"; df -h "$output/" | tail -1

mkdir -p logs
nohup timeout 17.75h julia --project=. -t 4 "$script" "$@" >> logs/fast_rotation_mechanism.log 2>&1 &
echo "Started (PID $!), log logs/fast_rotation_mechanism.log"
