#!/usr/bin/env bash
# Launches base_case_1km_resolution.jl in the background as two chains sharing the GPU, (P3_dT30, P5_dT30) and (P10_dT1, P10_dT30),
# logging to logs/res1km_<chain>.log. Extra arguments (ref, nz32) are passed to both chains.
#   experiments/tests/base_case_1km_resolution.sh            # 1° and 0.5° for all four planets
#   experiments/tests/base_case_1km_resolution.sh ref        # also 0.25°
# Rerunning skips finished runs and resumes cut-off ones.
set -euo pipefail

cd "$(dirname "$0")/../.."
script=experiments/tests/base_case_1km_resolution.jl
output="${EXO_OCEAN_OUTPUT:-output}"

nvidia-smi > /dev/null 2>&1 || { echo "nvidia-smi can't reach the GPU driver"; exit 1; }
if pgrep -f "julia.*base_case_1km_resolution.jl" > /dev/null; then
    echo "$script is already running:"; pgrep -af "julia.*base_case_1km_resolution.jl"; exit 1
fi
ls "$output"/checkpoints/gmredi_nowind_kv1e-4_Pr10_iteration*.jld2 > /dev/null 2>&1 ||
    echo "Warning: no gmredi_nowind_kv1e-4_Pr10 checkpoint in $output/checkpoints; P10_dT30, P5_dT30 and P3_dT30 will spin up from rest"

mkdir -p logs
declare -A chains=([a]="P3_dT30 P5_dT30" [b]="P10_dT1 P10_dT30")
for chain in a b; do
    # shellcheck disable=SC2086
    nohup julia --project=. -t 4 "$script" ${chains[$chain]} "$@" > "logs/res1km_$chain.log" 2>&1 &
    echo "Started ${chains[$chain]} $* (PID $!), log logs/res1km_$chain.log"
done
