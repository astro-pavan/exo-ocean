#!/bin/bash
# Runs the five sensitivity tests (experiments/tests/h100_test_*.jl) on CSD3 A100s as a job array: one A100 and 32 cores per test.
#   cd ~/exo-ocean && mkdir -p logs && sbatch scripts/csd3_sensitivity_tests.sbatch   (all five; Slurm needs logs/ to exist)
#   sbatch --array=1,4 scripts/csd3_sensitivity_tests.sbatch                (only some; indices as in TESTS below)
# First: precompile on a GPU node (docs/LLM/csd3_a100_guide.md, step 3) and copy the wind files and the 1 km checkpoint for the wind test.
# Rerunning a test skips its runs already marked ok, so a timed-out or failed task can simply be resubmitted.
#SBATCH -J sens
#SBATCH -A SHORTTLE-SL3-GPU
#SBATCH -p ampere
#SBATCH -N1
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out

TESTS=(grid_requirements warm_start vertical_mixing tracer_diffusivity wind)
TEST=${TESTS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vi cuda | paste -sd: || true)

echo "$(date '+%F %T')  $TEST on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
julia --project=. "experiments/tests/h100_test_$TEST.jl"
echo "$(date '+%F %T')  $TEST finished"
