#!/usr/bin/env bash
# Set up exo-ocean after cloning: Julia environment, GPU check, Python environment, output store, env.sh, smoke test.
# Safe to re-run. Usage: ./setup.sh [options]   (./setup.sh --help for the list)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

OUTPUT="${EXO_OCEAN_OUTPUT:-}"
PYTHON=""
THREADS="$(nproc 2>/dev/null || echo 4)"
CUDA_RUNTIME=""
DO_JULIA=1; DO_PYTHON=1; DO_GPU=1; DO_SMOKE=1; INSTALL_JULIA=0

usage() {
    cat <<EOF
Usage: ./setup.sh [options]

  --output DIR          Output store for simulation data (default: \$EXO_OCEAN_OUTPUT, else $REPO/output).
                        If $REPO/output doesn't exist yet, it is created as a symlink to DIR.
  --python PATH         Python 3.11 interpreter for the venv (default: python3.11, else python3).
  --threads N           Default Julia thread count for CPU runs, written to env.sh (default: $THREADS).
  --cuda-runtime X.Y    Pin the CUDA runtime CUDA.jl uses (e.g. 12.6). Chosen automatically for GPUs older than
                        compute capability 7.5, which CUDA 13 no longer supports.
  --install-julia       Install Julia 1.12 with juliaup if it isn't found.
  --cpu-only            Skip the GPU check and GPU smoke test.
  --skip-julia          Skip the Julia environment.
  --skip-python         Skip the Python environment.
  --no-smoke-test       Skip the smoke tests.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --cuda-runtime) CUDA_RUNTIME="$2"; shift 2 ;;
        --install-julia) INSTALL_JULIA=1; shift ;;
        --cpu-only) DO_GPU=0; shift ;;
        --skip-julia) DO_JULIA=0; shift ;;
        --skip-python) DO_PYTHON=0; shift ;;
        --no-smoke-test) DO_SMOKE=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die()  { printf '    ERROR: %s\n' "$*" >&2; exit 1; }

GPU_OK=0

# ---------------------------------------------------------------- 1. Prerequisites
step "Checking prerequisites"
case "$(uname -s)" in
    Linux|Darwin) info "OS: $(uname -s)" ;;
    *) die "Unsupported OS $(uname -s); use Linux or macOS." ;;
esac
command -v git >/dev/null || warn "git not found (only needed for updating the repo)."
depot="${JULIA_DEPOT_PATH:-$HOME/.julia}"; depot="${depot%%:*}"
mkdir -p "$depot" 2>/dev/null || true
free_gb=$(df -Pk "$depot" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
info "Julia depot: $depot (${free_gb:-?} GB free)"
[[ -n "${free_gb:-}" && "$free_gb" -lt 10 ]] && warn "Less than 10 GB free for the Julia depot. On HPC systems set JULIA_DEPOT_PATH to a scratch disk."

# ---------------------------------------------------------------- 2. Julia
if [[ $DO_JULIA -eq 1 ]]; then
    step "Julia"
    if ! command -v julia >/dev/null; then
        if [[ $INSTALL_JULIA -eq 1 ]]; then
            info "Installing Julia with juliaup..."
            curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12
            export PATH="$HOME/.juliaup/bin:$PATH"
        else
            die "julia not found. Install Julia 1.12 (https://julialang.org/downloads/, e.g. with juliaup) or re-run with --install-julia."
        fi
    fi
    jver=$(julia --version | awk '{print $3}')
    info "Found Julia $jver ($(command -v julia))"
    [[ "$jver" == 1.12.* ]] || die "This project needs Julia 1.12 (found $jver). With juliaup: juliaup add 1.12 && juliaup default 1.12"

    info "Installing pinned packages from Manifest.toml and precompiling (first time: ~15-20 min)..."
    julia --project="$REPO" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

    # ------------------------------------------------------------ 3. GPU
    if [[ $DO_GPU -eq 1 ]]; then
        step "GPU"
        if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
            gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
            cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
            info "NVIDIA GPU: $gpu (compute capability ${cc:-unknown})"
            if [[ -z "$CUDA_RUNTIME" && -n "$cc" ]] && awk -v c="$cc" 'BEGIN {exit !(c < 7.5)}'; then
                CUDA_RUNTIME="12.6"
                info "Compute capability $cc < 7.5 is not supported by CUDA 13; pinning the CUDA $CUDA_RUNTIME runtime."
            fi
            if [[ -n "$CUDA_RUNTIME" ]]; then
                julia --project="$REPO" -e "using CUDA; CUDA.set_runtime_version!(v\"$CUDA_RUNTIME\")"
                info "Wrote the CUDA runtime pin to LocalPreferences.toml (machine-specific, git-ignored)."
            fi
            if julia --project="$REPO" -e 'using CUDA; CUDA.functional() || exit(1); println("    CUDA ", CUDA.runtime_version(), " on ", CUDA.name(CUDA.device()))'; then
                GPU_OK=1
            else
                warn "CUDA isn't functional; simulations will run on the CPU. Check the NVIDIA driver, or try --cuda-runtime 12.6."
            fi
        else
            info "No NVIDIA GPU found; simulations will run on the CPU."
        fi
    fi
fi

# ---------------------------------------------------------------- 4. Python
VENV="$REPO/.venv"
if [[ $DO_PYTHON -eq 1 ]]; then
    step "Python"
    if [[ -z "$PYTHON" ]]; then
        if command -v python3.11 >/dev/null; then PYTHON=python3.11; else PYTHON=python3; fi
    fi
    command -v "$PYTHON" >/dev/null || [[ -x "$PYTHON" ]] || die "Python interpreter '$PYTHON' not found; pass --python /path/to/python3.11."
    pyver=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    info "Using $PYTHON (Python $pyver)"
    case "$pyver" in
        3.11) ;;
        3.1[2-9]) warn "requirements.txt is pinned and tested for Python 3.11; $pyver may need different versions." ;;
        *) die "Python $pyver is too old; this project needs Python 3.11 (pass --python)." ;;
    esac
    if [[ ! -x "$VENV/bin/python" ]]; then
        info "Creating virtual environment $VENV"
        "$PYTHON" -m venv "$VENV"
    else
        info "Reusing virtual environment $VENV"
    fi
    "$VENV/bin/pip" install --quiet --upgrade pip
    info "Installing requirements.txt (pinned packages + the analysis package)..."
    "$VENV/bin/pip" install --quiet -r "$REPO/requirements.txt"
    "$VENV/bin/pip" check >/dev/null || warn "pip check reported dependency conflicts; run $VENV/bin/pip check for details."
fi

# ---------------------------------------------------------------- 5. Output store
step "Output store"
if [[ -n "$OUTPUT" ]]; then
    mkdir -p "$OUTPUT"
    OUTPUT="$(cd "$OUTPUT" && pwd)"
    if [[ ! -e "$REPO/output" && ! -L "$REPO/output" ]]; then
        ln -s "$OUTPUT" "$REPO/output"
        info "Linked $REPO/output -> $OUTPUT"
    elif [[ "$(cd "$REPO/output" && pwd -P)" != "$(cd "$OUTPUT" && pwd -P)" ]]; then
        info "$REPO/output already exists and points elsewhere; EXO_OCEAN_OUTPUT in env.sh will select $OUTPUT."
    fi
else
    [[ -e "$REPO/output" ]] || mkdir -p "$REPO/output"
    OUTPUT="$(cd "$REPO/output" && pwd)"
fi
mkdir -p "$OUTPUT/ocean/checkpoints"
info "Simulation output goes to $OUTPUT/ocean"

# ---------------------------------------------------------------- 6. env.sh
step "Writing env.sh"
{
    echo "# Generated by setup.sh on $(date '+%F'); source it before working:  source env.sh"
    echo "export EXO_OCEAN_OUTPUT=\"$OUTPUT\""
    echo "export JULIA_PROJECT=\"$REPO\"            # plain 'julia' uses this repo's pinned environment"
    echo "export JULIA_NUM_THREADS=\"$THREADS\"      # threads for CPU runs (override with julia -t N)"
    if [[ -n "${JULIA_DEPOT_PATH:-}" ]]; then echo "export JULIA_DEPOT_PATH=\"$JULIA_DEPOT_PATH\""; fi
    if [[ -d "$HOME/.juliaup/bin" ]]; then echo "export PATH=\"$HOME/.juliaup/bin:\$PATH\""; fi
    if [[ -f "$VENV/bin/activate" ]]; then echo "source \"$VENV/bin/activate\""; fi
} > "$REPO/env.sh"
info "Wrote $REPO/env.sh (machine-specific, git-ignored)"

# ---------------------------------------------------------------- 7. Smoke tests
if [[ $DO_SMOKE -eq 1 && $DO_JULIA -eq 1 ]]; then
    step "Smoke tests"
    smoke_run() {  # $1 = run name, $2 = use_GPU
        EXO_OCEAN_OUTPUT="$OUTPUT" REPO="$REPO" julia --project="$REPO" -t "$THREADS" -e "
            include(joinpath(ENV[\"REPO\"], \"src\", \"ocean\", \"ocean_sim.jl\"))
            using Oceananigans.Units
            ocean_simulation(\"$1\", 10, 500meters, R_Earth, 30.0, 0.0, 1days; n_lat = 20, n_lon = 36, n_depth = 5,
                             use_GPU = $2, output_interval = 6hours)
        " > "$OUTPUT/ocean/$1.log" 2>&1 || { tail -20 "$OUTPUT/ocean/$1.log" >&2; die "Smoke run $1 failed (log: $OUTPUT/ocean/$1.log)."; }
        info "$1: ok"
    }
    info "Running a 1-day CPU simulation on a tiny grid..."
    smoke_run setup_smoke_cpu false
    if [[ $GPU_OK -eq 1 ]]; then
        info "Running the same on the GPU..."
        smoke_run setup_smoke_gpu true
    fi
    if [[ $DO_PYTHON -eq 1 ]]; then
        info "Reading the output with the Python analysis package..."
        EXO_OCEAN_OUTPUT="$OUTPUT" "$VENV/bin/python" -c "
import contextlib, io, warnings
warnings.filterwarnings('ignore')
from analysis.paths import OCEAN_DIR
from analysis.simulation_reader import SimulationData
with contextlib.redirect_stdout(io.StringIO()):
    s = SimulationData(str(OCEAN_DIR / 'setup_smoke_cpu.nc'))
print('    analysis read setup_smoke_cpu.nc: T', s.T.shape)
" || die "Python could not read the smoke-test output."
    fi
    rm -f "$OUTPUT"/ocean/setup_smoke_*.nc "$OUTPUT"/ocean/setup_smoke_*.log
fi

# ---------------------------------------------------------------- Done
step "Setup complete"
cat <<EOF
    Start each session with:   source env.sh
    Run an experiment:         julia experiments/tests/test_ocean.jl
    Run the analysis:          python analysis/coriolis_jet_analysis.py
    Interactive viewer:        bokeh serve --show analysis/visualisation.py --args \$EXO_OCEAN_OUTPUT/ocean/<run>.nc
    GPU: $([[ $GPU_OK -eq 1 ]] && echo "available (runs with use_GPU=true use it)" || echo "not available (runs fall back to the CPU)")
EOF
