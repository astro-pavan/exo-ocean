#!/usr/bin/env bash
# Launches p3_resolution_test.jl in the background on this machine's GPU, logging to logs/p3_resolution_test.log. LIMIT sets the time limit (default 96h).
# Tests run in priority order, so the limit cuts the least important. Rerunning skips finished tests, resumes cut-off ones and appends to the log.
#   experiments/tests/p3_resolution_test.sh [test name filters, all of which must match]
set -euo pipefail

cd "$(dirname "$0")/../.."
script=experiments/tests/p3_resolution_test.jl

nvidia-smi > /dev/null 2>&1 || { echo "nvidia-smi can't reach the GPU driver"; exit 1; }
if pgrep -f "julia.*(p3_resolution_test|fast_rotation_[a-z]*).jl" > /dev/null; then
    echo "Already running:"; pgrep -af "julia.*(p3_resolution_test|fast_rotation_[a-z]*).jl"; exit 1
fi
output="${EXO_OCEAN_OUTPUT:-output}"
ls "$output"/checkpoints/fastdiag_heat_P3_cap0.5_iteration*.jld2 > /dev/null 2>&1 ||
    echo "Warning: no fastdiag_heat_P3_cap0.5 checkpoint in $output/checkpoints; the ΔT = 30 K tests will be skipped"
echo "Free disk space for the output (these runs can reach ~10 GB):"; df -h "$output/" | tail -1

mkdir -p logs
nohup timeout "${LIMIT:-96h}" julia --project=. -t 4 "$script" "$@" >> logs/p3_resolution_test.log 2>&1 &
echo "Started (PID $!), log logs/p3_resolution_test.log"
