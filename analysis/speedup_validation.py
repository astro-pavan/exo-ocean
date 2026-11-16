"""Compare each run of experiments/tests/speedup_validation.jl with its control over the last 3 saved years. A speed-up passes when every
difference is no larger than the twin controls' (P = 10 d, CFL 0.2 vs 0.199), or than FLOOR % (CORE_FLOOR m for the core depth)
where that is smaller.
    python analysis/speedup_validation.py [output folder]"""
import contextlib
import io
import os
import re
import sys
import warnings

import numpy as np
import tomllib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.jet_stop import yearly_jet
from analysis.simulation_reader import SimulationData

FLOOR = 2    # %, the smallest tolerance used when the twins happen to agree more closely
CORE_FLOOR = 15    # m, the same for the core depth, compared in metres since a near-surface core makes percentages meaningless
NAMES = ('jet speed', 'jet width', 'jet height', 'core depth', 'thermocline', 'eq. thermocline', 'mean T', 'deep T')
CORE = NAMES.index('core depth')
GROUPS = [('main_P10_control', ['main_P10_twin', 'main_P10_tkeoff', 'main_P10_tkeoff_cfl0.3', 'main_P10_f32']),
          ('main_P10_tkeoff', ['main_P10_tkeoff_f32']),
          ('main_P3_control', ['main_P3_tkeoff', 'main_P3_f32']),
          ('main_P3_tkeoff', ['main_P3_tkeoff_f32']),
          ('spinup_kv1e-5_weno', ['spinup_kv1e-5_weno_f32']),
          ('spinup_kv1e-2_explicit', ['spinup_kv1e-2_implicit'])]
STABILITY = ['rest_1deg_f32', 'rest_zonal_f32', 'corner_kv1e-5_f32', 'corner_kv1e-2_f32', 'corner_dT1_rest_f32', 'corner_P100_rest_f32',
             'corner_P3_rest_f32']    # Float32 runs with no Float64 partner: they only have to finish without a NaN
LOG = os.path.join(os.path.dirname(__file__), '..', 'logs', 'speedup_validation.log')
START = re.compile(r'^(\d+) .*Validation spdval_(\S+):')
PROGRESS = re.compile(r'^(\d+) .*Time: [\d.]+ \w+, Δt: ([\d.]+) (\w+),')
UNITS = {'seconds': 1 / 3600, 'minutes': 1 / 60, 'hours': 1, 'hour': 1, 'days': 24, 'day': 24}


def unit(i):
    return ' m' if i == CORE else '%'


def state(directory, run):
    """The NAMES quantities over the last 3 saved years, or None if the run has no output"""
    path = os.path.join(directory, f'spdval_{run}.nc')
    if not os.path.exists(path):
        return None
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        d = SimulationData(path, time=slice(-3, None))
        speed, core, width, height, _ = d.jet_properties()
        thermocline, equatorial, deep_T = d.deep_circulation_properties()
    return np.array([speed, width, height, core, thermocline, equatorial, d.volume_mean(d.T), deep_T])


def progress(directory, run):
    """Why a run has no 3D output yet: not started, still running, or stopped early (3D output covers only its last 3 planned years)"""
    path = os.path.join(directory, f'spdval_{run}_parameters.toml')
    if not os.path.exists(path):
        return 'not run yet'
    with open(path, 'rb') as f:
        finished = tomllib.load(f).get('finished')
    if finished is None:
        return 'running or cut off'
    return f'stopped by {finished["stopped_by"]} at {finished["model_time"] / (365 * 86400):.1f} yr with no 3D output'


def finish(directory, run):
    """How a run ended: NaN, or clean with its stop reason and model years"""
    path = os.path.join(directory, f'spdval_{run}_parameters.toml')
    if not os.path.exists(path):
        return 'not run yet'
    with open(path, 'rb') as f:
        finished = tomllib.load(f).get('finished')
    if finished is None:
        return 'running or cut off'
    years = finished['model_time'] / (365 * 86400)
    return 'NaN' if finished['stopped_by'] == 'nan' or not np.isfinite(years) else f'clean, {years:.1f} yr ({finished["stopped_by"]})'


def years_run(directory, run):
    path = os.path.join(directory, f'spdval_{run}_section.nc')
    return yearly_jet(path) if os.path.exists(path) else (np.array([]), np.empty((0, 3)))


def speeds(path=LOG):
    """{run: (steps/s, GPU min per model yr)} from the timestamped log, skipping each run's first 3 progress lines (compilation)"""
    found, current = {}, None
    if not os.path.exists(path):
        return {}
    with open(path, errors='replace') as f:
        for line in f:
            m = START.match(line)
            if m:
                current = m.group(2)
                found[current] = []
            elif 'Spinning up' in line:    # a donor spin-up between tests isn't the previous test's
                current = None
            elif current and (m := PROGRESS.match(line)):
                found[current].append((int(m.group(1)), float(m.group(2)) * UNITS.get(m.group(3), np.nan)))
    out = {}
    for run, lines in found.items():
        lines = lines[3:]
        if len(lines) > 2 and lines[-1][0] > lines[0][0]:
            rate = 100 * (len(lines) - 1) / (lines[-1][0] - lines[0][0])
            out[run] = (rate, 8760 / np.median([dt for _, dt in lines]) / rate / 60)
    return out


def main(directory):
    twin = None
    rates = speeds()
    for control, tests in GROUPS:
        reference = state(directory, control)
        print(f'\n{control}: ' + (progress(directory, control) if reference is None else ', '.join(f'{n} {v:.4g}' for n, v in zip(NAMES, reference))))
        if reference is None:
            continue
        for test in tests:
            s = state(directory, test)
            if s is None:
                print(f'  {test:30s} {progress(directory, test)}')
                continue
            difference = 100 * (s / reference - 1)
            difference[CORE] = s[CORE] - reference[CORE]
            floor = np.full(len(NAMES), float(FLOOR))
            floor[CORE] = CORE_FLOOR
            if test == 'main_P10_twin':
                twin = np.maximum(np.abs(difference), floor)
            tolerance = twin if twin is not None else floor
            failed = [n for n, d, t in zip(NAMES, difference, tolerance) if abs(d) > t]
            verdict = 'noise level' if test == 'main_P10_twin' else ('PASS' if not failed else 'FAIL: ' + ', '.join(failed))
            print(f'  {test:30s} ' + ', '.join(f'{n} {d:+.1f}{unit(i)}' for i, (n, d) in enumerate(zip(NAMES, difference))) + f'  [{verdict}]')
        print('  yearly jet speed (m/s):')
        for run in [control] + tests:
            years, m = years_run(directory, run)
            print(f'    {run:30s} ' + ' '.join(f'{v:.3f}' for v in m[:, 0]))
    print('\nFloat32 stability runs (no Float64 partner):')
    for run in STABILITY:
        print(f'  {run:30s} {finish(directory, run)}')
    if rates:
        print('\nSpeed from the log (steps/s, GPU min per model yr at the run\'s median Δt):')
        for run, (rate, minutes) in rates.items():
            print(f'  {run:30s} {rate:7.1f} {minutes:7.2f}')
    if twin is not None:
        print('\nTolerances from the twin controls: ' + ', '.join(f'{n} {t:.1f}{unit(i)}' for i, (n, t) in enumerate(zip(NAMES, twin))))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'output'))
