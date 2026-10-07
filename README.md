# exo-ocean

A study of tidally locked oceans, their superrotating equatorial jets and their heat transport.

## Repository layout

```
src/ocean/          Julia ocean model (Oceananigans): ocean_sim.jl, constants, emulator wind forcing.
src/climate/        Python CLERO climate emulator: matches an ocean run to a planet and writes its winds.
experiments/sweeps/ Julia parameter sweeps (rotation, buoyancy forcing).
experiments/tests/  Julia test and diagnostic runs (deep equilibrium, resolution, warm starts, emulator winds).
analysis/           Python package that reads simulation output and makes figures.
figures/            Generated plots (git-ignored).
Project.toml        Julia dependencies; Manifest.toml pins the exact versions.
requirements.txt    Python dependencies (pinned), plus the analysis package itself.
pyproject.toml      Makes analysis/ an installable package.
output/             Simulation output store (git-ignored; a directory or a symlink to one).
```

## Quick start

```bash
git clone <repo-url> exo-ocean && cd exo-ocean
./setup.sh --output /path/to/large/disk/exo_ocean_sims   # see ./setup.sh --help
source env.sh                                            # at the start of each session
```

`setup.sh` installs the pinned Julia packages and checks the GPU, creates the Python environment (`.venv`), sets up the output store, writes `env.sh`, and runs a short smoke test. It needs Julia 1.12 (or `--install-julia` to install it with juliaup) and Python 3.11. It's safe to re-run. The manual steps it performs are below.

## Setup (manual)

**Julia** (1.12):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Always run with `--project=.` so the pinned versions in `Manifest.toml` are used. A CUDA GPU is used automatically when one is available; otherwise runs fall back to the CPU.

**Python** (3.11), from the repo root:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Output location.** By default, output goes to `output/` in the repo (create it, or symlink it to a large disk). To use another location, set `EXO_OCEAN_OUTPUT`, which both the Julia runs and the Python analysis respect:

```bash
export EXO_OCEAN_OUTPUT=/path/to/large/disk/exo_ocean_sims
```

Ocean runs write to `$EXO_OCEAN_OUTPUT/ocean/` (checkpoints in `ocean/checkpoints/`).

## Running

Julia experiments (from any directory):

```bash
julia --project=. experiments/sweeps/coriolis_sweep.jl
julia --project=. experiments/tests/deep_equilibrium.jl 1                     # 1 km ocean to deep equilibrium (GPU)
julia --project=. -t 16 experiments/tests/deep_equilibrium.jl 1 --res 4 --cpu  # CPU runs: set the thread count with -t
```

Python analysis (from any directory, with the venv active):

```bash
python analysis/coriolis_jet_analysis.py
bokeh serve --show analysis/visualisation.py --args output/ocean/<run>.nc    # interactive 3D viewer
```

Emulator winds for a wind-forced run:

```bash
python src/climate/climate_emulator.py solve --P_rot 10 --T_day 30 --T_night 0 --P0 3 --out output/winds/P10_Td30_Tn0_P3.nc
```
