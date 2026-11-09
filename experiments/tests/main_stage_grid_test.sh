#!/usr/bin/env bash
# Runs main_stage_grid_test.jl in the background on the RTX 2000, logging to logs/main_stage_grid_test.log, and stops it after 12 hours in total.
# Tests run in priority order, so the limit cuts the least important. Rerunning skips finished tests, resumes cut-off ones and appends to the log.
#   experiments/tests/main_stage_grid_test.sh [test name filters, all of which must match]
set -euo pipefail

cd "/home/pavan/exo-ocean"
script=experiments/tests/main_stage_grid_test.jl
limit=12h

nvidia-smi -L 2> /dev/null | grep -q "RTX 2000" || { echo "No RTX 2000 on this machine (nvidia-smi -L lists: $(nvidia-smi -L 2>&1 | head -3))"; exit 1; }
if pgrep -f "julia.*(main_stage_grid_test|p3_resolution_test|fast_rotation_[a-z]*).jl" > /dev/null; then
    echo "Already running:"; pgrep -af "julia.*(main_stage_grid_test|p3_resolution_test|fast_rotation_[a-z]*).jl"; exit 1
fi

output="${EXO_OCEAN_OUTPUT:-output}"
for delta in 30 1; do
    donor="fastdiag_p3res_dT${delta}_spinup"
    ls "$output"/checkpoints/"${donor}"_iteration*.jld2 > /dev/null 2>&1 || { echo "No checkpoint of $donor in $output/checkpoints"; exit 1; }
    [ -f "$output/fastdiag_p3res_dT${delta}_0.5deg_section.nc" ] || echo "Note: no 0.5° reference run for dT${delta} in $output, so the comparison needs it copied here"
done
echo "Free disk space for the output (these runs can reach ~6 GB):"; df -h "$output/" | tail -1

mkdir -p logs
nohup timeout "$limit" julia --project=. -t 4 "$script" "$@" >> logs/main_stage_grid_test.log 2>&1 &
echo "Started (PID $!) with a $limit limit, log logs/main_stage_grid_test.log"
