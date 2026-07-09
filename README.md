# exo-ocean

A study on tidally locked oceans and their heat transport capacity.

## Repository layout

```
src/          Julia model code — the simulation "engine" (Oceananigans ocean,
              SpeedyWeather atmosphere, shared constants & custom physics).
experiments/  Julia driver scripts that call src/ to run sweeps and tests.
analysis/     Python scripts that read simulation output and produce figures.
figures/      Generated plots (PNGs, git-ignored).
Project.toml  Julia environment (pinned dependencies).

output/   Symlink to the NetCDF output store.
```

## Running

Julia experiments (from the repo root, so the `exo_ocean_sims/` output path resolves):

```bash
julia --project=. experiments/coriolis_sweep.jl
```

Python analysis (from `analysis/`, so the sibling module imports resolve):

```bash
cd analysis && /data/pt426/big-venv/bin/python coriolis_jet_analysis.py
```
