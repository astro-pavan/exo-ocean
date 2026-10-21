#!/usr/bin/env bash
# Launches the three chains of wind_equilibrium_rtx2000.jl in the background, one per wind case, logging to logs/wind_equilibrium_<case>.log.
#   experiments/tests/wind_equilibrium_rtx2000.sh
# Rerunning skips finished runs and resumes cut-off ones.
set -euo pipefail

cd "$(dirname "$0")/../.."
export EXO_EMULATOR_PYTHON="${EXO_EMULATOR_PYTHON:-$HOME/ocean-venv/bin/python}"

[ -x "$EXO_EMULATOR_PYTHON" ] || { echo "No Python at $EXO_EMULATOR_PYTHON (set EXO_EMULATOR_PYTHON)"; exit 1; }
nvidia-smi > /dev/null 2>&1 || { echo "nvidia-smi can't reach the GPU driver"; exit 1; }
if pgrep -f "julia.*wind_equilibrium_rtx2000.jl" > /dev/null; then
    echo "wind_equilibrium_rtx2000.jl is already running:"; pgrep -af "julia.*wind_equilibrium_rtx2000.jl"; exit 1
fi

mkdir -p logs
for case in nowind 2 10; do
    nohup julia --project=. -t 4 experiments/tests/wind_equilibrium_rtx2000.jl "$case" > "logs/wind_equilibrium_$case.log" 2>&1 &
    echo "Started $case (PID $!), log logs/wind_equilibrium_$case.log"
done
