# Parameterising the CLERO surface wind on tidally locked planets

**Goal:** a simple closed-form function that takes climate inputs (day–night contrast ΔT, surface pressure P0, rotation period P_rot) and returns the near-surface wind, with an error about equal to the UM vs ExoCAM difference in the CLERO emulator, using as few parameters as possible.

## Summary

- **The target is not fully reached.** The best small functions are close to UM vs ExoCAM on the ocean forcing metrics, but not on the wind pattern. On all periods, wind error is about 1.3–1.4× and direction error about 1.5–1.7× the same-climate UM vs ExoCAM difference.
- **Individual planets can reach the target.** A 10-parameter form fitted per planet scores 0.345 / 0.240 / 15.4° (wind / speed / direction), against 0.356 / 0.268 / 12.9° for UM vs ExoCAM. The remaining gap is mostly in expressing how the wind varies with climate, not in the shape of the form.
- **Candidates for a final function:**
  - **22 numbers:** the most accurate after pruning (0.457 / 0.341 / 19.5°), but it needs refitting with the polar fix in section 8.
  - **14–16 numbers (section 7):** 0.486–0.504 wind error. The later and more complex forms do not clearly beat these at the small end.
- **No final function has been chosen.** None of the functions developed here has been scored on the held-out planets (see "Caveats").

## 1. Set-up

### Data

- **Emulator:** `clero==0.1.3`, 32 Gaussian latitudes × 64 longitudes, `GCM="exocam"` and `GCM="um"`.
- **Sweep:** 2,449 ExoCAM planets regenerated from the defaults in `experiments/sweeps/parametric_wind_sweep.py` (seed 0), because the original sweep files were not on this machine. It may not match other copies exactly.
- **Splits:** `jwf.split` divides planets into training and held-out halves. The training half is then split into FIT (612 planets) and VAL (612). Unless stated otherwise, every score below is the median over VAL.

### Benchmarks

| UM vs ExoCAM comparison | Wind | Speed | Direction | `tau_conv` | `tau_prof` | `ustar3` | `ustar3_map` | `ustar3` bias |
|---|---|---|---|---|---|---|---|---|
| Same climate (re-solved planets) | 0.356 | 0.268 | 12.9° | 0.240 | 0.415 | 0.369 | 0.570 | +24% |
| Same inputs | 0.392 | 0.276 | 14.2° | 0.219 | 0.443 | 0.289 | 0.582 | +16% |

- **Same climate** inverts each planet with `solve_planet` so that UM matches the ExoCAM climate. It is the fairer limit for a function whose input is climate.
- **Coverage:** UM solved 234 of 300 planets, with failures concentrated at high P0 (46% above 4 bar) and at fast rotation (62% at 4–8 d, leaving only 11 planets). The ExoCAM control solved 300 of 300 with a median error of about 0.
- **Fast rotators (P_rot < 15 d):** the same-climate benchmark is 0.585 wind and 24.7° direction, so the target is much looser there.

### Metrics

- **Wind:** relative RMS vector error over |lat| ≤ 80°.
- **Speed:** relative RMS speed error.
- **Direction:** angle error weighted by area × emulator speed. The conclusions do not depend on this choice: the climate function stayed at 1.6–1.7× UM vs ExoCAM under area-only, speed and speed² weighting.
- **Ocean metrics:** `tau_conv`, `tau_prof`, `ustar3` and `ustar3_map` from `experiments/tests/converging_wind_test.py` (Large & Yeager bulk stress).
- **Rescale:** later functions multiply U_0 by a factor chosen on FIT to zero the median `tau_conv` bias. It scales the core and the rotational terms, but not the zonal jet.

Notation: φ is latitude, λ is longitude from the substellar point, θ is angular distance from it (cos θ = cos φ cos λ), α is azimuth around it, p = P0 / 2 bar, L = ln(ΔT / 40 K).

## 2. Existing seven-parameter scheme

Per-planet fits of `parametric_wind` (stress-fitted) against UM vs ExoCAM at the same inputs, over all 2,449 planets:

| Metric | Per-planet fit | UM vs ExoCAM | Median ratio |
|---|---|---|---|
| Relative stress | 0.385 | 0.459 | 0.83 |
| Relative wind | 0.464 | 0.375 | 1.09 |
| `tau_conv` | 0.056 | 0.214 | 0.30 |
| `tau_prof` | 0.247 | 0.440 | 0.59 |
| `ustar3` | 0.152 | 0.285 | 0.60 |
| `ustar3_map` | 0.336 | 0.592 | 0.62 |

The per-planet fit is well inside the GCM difference on ocean metrics but slightly worse on the wind itself. The existing joint fit (`simple`, 16 coefficients) was about level overall, but much worse at 60–120 d (`tau_conv` ratio 4.8).

## 3. Is there a regime change with rotation?

The transition is real but gradual, mostly between about 5 and 25 d, and the flow shape is then roughly constant to 120 d. One smooth switch is justified, for example s = 1 / (1 + (P_rot / P_t)²).

| P_rot (d) | θ_E (rad) | θ_W (rad) | f | U_p (m/s) |
|---|---|---|---|---|
| 4.6 | 1.19 | 0.18 | 0.71 | 9.75 |
| 10.8 | 0.69 | 0.33 | 0.54 | 9.72 |
| 19 | 0.54 | 0.44 | 0.25 | 9.42 |
| 25 | 0.49 | 0.42 | 0.17 | 9.51 |
| 59 | 0.43 | 0.46 | 0.16 | 10.74 |
| 104 | 0.44 | 0.51 | 0.19 | 11.15 |

These are ExoCAM medians, not controlled for ΔT or P0. CLERO is a smooth emulator, so it could blur a sharper change present in the underlying GCMs.

## 4. Stress-based functions

These fits targeted stress and `u*³`.

| Function | Fitted | `tau_conv` | `tau_prof` | `ustar3` | `ustar3_map` |
|---|---|---|---|---|---|
| `lean11`, single function | 11 | 0.94 | 1.07 | 0.94 | 1.02 |
| Two functions, boundary at 12 d, rescaled | 11 | 0.84 | 1.08 | 0.84 | 1.05 |
| Two functions, boundary at 15 d, rescaled | 10 | 0.83 | 1.07 | 0.82 | 1.06 |
| **Fast (P < 15 d) and slow (P ≥ 10 d) blended over 10–15 d** | 10 | **0.80** | **1.06** | **0.80** | **1.05** |

Values are ratios to the same-climate benchmark.

- **Main findings:** five coefficients per regime are enough on the ocean metrics. A U_p rescale removes most of the low bias. The sharp boundary caused a 27–29% wind jump, which the blend removes.
- **Remaining problem:** `ustar3` was about 30% low above 4 bar.
- **Why this approach was abandoned:** side-by-side plots looked visibly different from the emulator. A pattern can match stress and `u*³` with the wrong wind, and refitting to the wind vectors showed the form itself was limiting.

Fitting the actual winds with the same form (7 + 7 coefficients, blended) gave 0.528 / 0.387 / 26°, with winds 25% too weak. Even 7 free parameters per planet gave only 0.441 / 0.302 / 20°, so no choice of coefficients could reach the target. The form had no rotational part.

## 5. Adding the rotational flow

Tests below are per-planet fits for fast rotators (P_rot < 15 d), wind / speed / direction.

| Form | Parameters | Result |
|---|---|---|
| UM vs ExoCAM, same climate | | 0.585 / 0.372 / 25° |
| Converging core only | 5 | 0.624 / 0.469 / 41° |
| Converging core + 2 rotational modes (`xz`, `psi33`) | 7 | 0.505 / 0.361 / 22° |
| Matsuno–Gill alone | 4 | 0.903 / 0.715 / 70° |
| Matsuno–Gill + core + jet | 7 | 0.566 / 0.407 / 29° |

**Matsuno–Gill does not describe these surface winds.** Its flow inside the heated region is poleward, while the emulator's is equatorward, and fitted values sat on their limits.

**Visual inspection of the emulator** (`figures/emulator_wind_*.png`) showed a stable converging core, plus two features that fade as rotation slows: a long eastern equatorial lobe and eastward zonal jets near 60–75°. A Gaussian zonal jet is the most useful term from this stage:

u_jet = J exp(−(|φ| − φ_J)² / 2w²)

Per-planet results over all periods:

| Form | Parameters | Wind | Speed | Direction | Fast wind / direction |
|---|---|---|---|---|---|
| Core 4 + jet | 7 | 0.426 | 0.296 | 20° | 0.568 / 32° |
| Core 4 + jet + `psi33` | 8 | 0.398 | 0.291 | 17° | 0.537 / 28° |
| Core 4 + jet + east jet + dipole + shift | 11 | 0.388 | 0.272 | 18.5° | 0.507 / 27.5° |

**Where the error sits** (8-coefficient climate function, `figures/wind_error_maps.png`): the largest relative errors are on the nightside (wind 0.91, against 0.76 for UM vs ExoCAM), at high latitude (0.62 against 0.52) and on the flanks (0.54 against 0.37), not in the main inflow lobes. The form forces the nightside speed towards zero, which leaves it about 78% too low.

## 6. Climate functions and pruning

Fitting each form as a function of climate loses about 0.1 in wind error relative to per-planet fits.

| Function | Numbers in formula | Wind | Speed | Direction | `tau_conv` | `tau_prof` | `ustar3` | `ustar3_map` |
|---|---|---|---|---|---|---|---|---|
| 8 fitted + 9 fixed, core + jet | 17 | 0.519 | 0.377 | 23° | 0.197 | 0.438 | 0.339 | 0.614 |
| All 11-parameter terms | 27 | 0.485 | 0.369 | 22.4° | 0.201 | 0.483 | 0.329 | 0.645 |
| Pruned to 11 fitted | 25 | 0.487 | 0.372 | 22.2° | 0.207 | 0.492 | 0.330 | 0.643 |
| `psi33` replaces east jet, no shift, jet group, f = f_f s | 16 | 0.486 | 0.374 | 21.5° | 0.205 | 0.446 | 0.336 | 0.627 |
| As above, new profile, no dipole | 14 | 0.504 | 0.386 | 21.8° | 0.203 | 0.450 | 0.332 | 0.635 |

Fixed values count as numbers in the formula. Rounding a coefficient reduces the fitted count, not the size of the formula.

## 7. The 14-number function

With s = 1 / (1 + (P_rot / 15 d)²) and R = (ΔT/40)^0.2 p^−0.466, including the rescale:

```
u_core = −M sin λ / sin θ
v_core = −M sin φ cos λ / sin θ
M      = U_p · S(θ; θ_p) · (1 − f sin²α)
θ_p    = θ_E cos²(α/2) + θ_W sin²(α/2)

U_p = 10.30 R
θ_E = [0.370 + (0.734 − 0.370) s] p^−0.4
θ_W = 0.370 p^−0.4
f   = 1.01 s
S   = sin(π θ / 2θ_p)                         for θ < θ_p
S   = cos^2.98( π (θ − θ_p) / 2(π − θ_p) )    for θ ≥ θ_p

Zonal jet: J = 1.08 m/s × [ (ΔT/40)³ p / (P_rot / 8 d) ]^(2/3),  at 70°, width 15°
Rotation:  ψ33 mode with amplitude 1.57 m/s × s R

ψ33 = sin φ (3 − 3 cos²φ cos²λ − 4 sin²φ),  u = −∂ψ/∂φ,  v = ∂ψ / (cos φ ∂λ),
      divided by its area-weighted RMS speed over |lat| ≤ 80°
```

- **Against the target:** wind is 1.42× the same-climate difference (1.23× paired at the same inputs) and direction is 1.68×. `ustar3` is 16% low.
- **Interpretation:** the jet group (ΔT³ p / P_rot)^(2/3) is a numerical match to the fitted exponents, not a physical derivation.
- **The 16-number version** keeps the original sech-based profile and the dipole term, with `psi33` replacing the east jet and no shift.

### Comparison with the emulator, component by component

Using a Helmholtz decomposition, in units of each planet's RMS wind:

| Part of the wind | Emulator energy | Function energy | Function error² | UM vs ExoCAM error² |
|---|---|---|---|---|
| Divergent | 0.76 | 0.75 | 0.077 | 0.039 |
| Rotational, zonal mean | 0.05 | 0.02 | 0.043 | 0.021 |
| Rotational, eddy | 0.14 | 0.01 | 0.111 | 0.084 |

The core's amplitude and the jet's shape were sound. The problems were: the `f` factor suppressing north–south inflow at fast rotation (peak about 0.3, against 0.95 in ExoCAM at 4–8 d), a missing weak tail in the fall-off, a missing gyre pair east of the core, and a jet of the wrong sign beyond about 50 d. These findings led to the next form.

## 8. Latest form

```
Core:     as above with longitude shifted by λ0 and, beyond θ_p, with x = (θ − θ_p) / (π − θ_p),
          S = (1 − b) cos^q(π x / 2) + b cos(π x / 2),   q = q_0 + dq sin²α
Jet:      u = J exp(−(|φ| − φ_J)² / 2w²)
Gyre:     ψ = G (φ/φ_g) exp(½(1 − (φ/φ_g)²)) exp(−(λ − λ_g)² / 2σ²)
ψ33:      amplitude A
Climate:  s = 1 / (1 + (P_rot/P_t)²),  R = exp(b_U L + c_U ln p),  U_p = U_0 R,  G = G_0 s R,  b = b_s (1 − s)
          J = J_0 (P_rot/8)^a_J p^c_J (ΔT/40)^d_J / (1 + (P_rot/P_J)⁴)
          θ_E, θ_W scale as p^c_t;  q_0, dq, f, λ0, A blend slow → fast with s
```

### Per-planet results

| Form | Parameters | All periods | Fast: wind / direction | Slow: wind / direction |
|---|---|---|---|---|
| `f`, `ψ33` | 7 | 0.403 / 0.290 / 18.5° | 0.539 / 29.9° | 0.322 / 13.4° |
| Replace `f` with direction-dependent fall-off | 7 | 0.399 / 0.273 / 18.2° | 0.543 / 27.8° | 0.323 / 13.4° |
| + tail | 8 | 0.388 / 0.258 / 17.6° | 0.542 / 27.8° | 0.310 / 13.1° |
| + shift | 9 | 0.363 / 0.254 / 16.7° | 0.518 / 25.3° | 0.277 / 11.8° |
| Swap `ψ33` for gyre pair, fixed shape | 9 | 0.376 / 0.263 / 17.9° | 0.493 / 25.4° | 0.286 / 12.2° |
| Gyre pair + `ψ33` | 10 | 0.345 / 0.240 / 15.4° | 0.478 / 23.5° | 0.264 / 11.1° |
| Gyre pair, free shape, no `ψ33` | 12 | 0.332 / 0.227 / 14.6° | 0.415 / 19.7° | 0.248 / 9.9° |

The fitted core shift is about 5° west at all rotation rates, contrary to the eastward offset expected at fast rotation from the emulator cross-sections. In the 12-parameter form, parameters hit their limits for 22–26% of planets (`φ_g`, `λ_g`, `dq`), so part of its gain may be compensation.

### Pruning the climate function

Starting from 31 coefficients, the ladder dropped or tied whichever term cost the least, then rounded values only when nothing else could be removed.

| Numbers | Last step | Wind | Speed | Direction | Fast: wind / direction | `tau_conv` | `tau_prof` | `ustar3` | `ustar3_map` | `ustar3` bias |
|---|---|---|---|---|---|---|---|---|---|---|
| 31 | All coefficients | 0.448 | 0.341 | 19.3° | 0.556 / 27.2° | 0.192 | 0.464 | 0.311 | 0.612 | −16% |
| 26 | Tie `ψ33` | 0.450 | 0.341 | 19.1° | 0.549 / 26.3° | 0.192 | 0.459 | 0.322 | 0.605 | −17% |
| **22** | **Tie `q`** | **0.457** | **0.341** | **19.5°** | **0.552 / 26.7°** | 0.199 | 0.454 | 0.327 | 0.602 | −19% |
| 21 | Drop direction fall-off | 0.461 | 0.343 | 19.3° | 0.560 / 26.4° | 0.209 | 0.471 | 0.343 | 0.624 | −17% |
| 19 | Drop `f` | 0.474 | 0.345 | 19.2° | 0.596 / 26.5° | 0.203 | 0.478 | 0.352 | 0.621 | −8% |
| 18 | Tie slow θ_W = θ_E | 0.480 | 0.347 | 19.4° | 0.607 / 26.8° | 0.195 | 0.478 | 0.344 | 0.617 | −10% |
| 17 | Drop shift | 0.483 | 0.344 | 20.3° | 0.607 / 27.3° | 0.197 | 0.455 | 0.345 | 0.609 | −11% |
| 16 | Drop tail | 0.509 | 0.407 | 22.5° | 0.642 / 29.2° | 0.267 | 0.499 | 0.394 | 0.655 | −18% |
| 12 | Drop gyre | 0.577 | 0.437 | 22.9° | 0.863 / 36.8° | 0.197 | 0.486 | 0.446 | 0.840 | +28% |

**Terms that are cheap to remove (31 → 22):** `ψ33`, the jet cutoff, the separate fast and slow values of θ_E, θ_W, q, dq and the shift, and the separate jet P0 and rotation exponents. The tail and gyre pair are the most costly to remove.

**Against UM vs ExoCAM by band (22 numbers):**

| P_rot | Wind ratio, same climate | Direction ratio, same climate | Wind ratio, paired at same inputs | Mean speed bias |
|---|---|---|---|---|
| 4–8 d | 1.16 | 1.02 | 0.96 | −14% |
| 8–15 d | 0.89 | 1.09 | 1.01 | −20% |
| 15–30 d | 1.28 | 1.53 | 1.07 | −16% |
| 30–60 d | 1.01 | 0.88 | 1.02 | −5% |
| 60–120 d | 1.22 | 1.17 | 1.43 | −16% |

Overall, the 22-number ratios are 1.29 (wind) and 1.51 (direction) against the same-climate benchmark. The paired wind ratio against same inputs is 1.09. The 15–30 d band is worst, suggesting a single fast-to-slow transition is too simple.

### The polar defect and follow-up tests

The gyre's streamfunction does not vanish at the pole, so the ladder functions with a gyre have spurious strong meridional flow near ±80° (visible in `figures/wind_form4.png`). Multiplying ψ by cos²φ removes it by construction. These fits ran 40 iterations, so they may not be fully converged:

| Function | Numbers | Wind | Speed | Direction | Fast: wind / direction | `tau_conv` | `tau_prof` | `ustar3` | `ustar3_map` | `ustar3` bias |
|---|---|---|---|---|---|---|---|---|---|---|
| Tapered gyre + shift | 18 | 0.479 | 0.356 | 20.9° | 0.637 / 29.0° | 0.197 | 0.455 | 0.326 | 0.610 | −13% |
| Tapered gyre | 17 | 0.486 | 0.361 | 21.7° | 0.640 / 29.3° | 0.204 | 0.450 | 0.336 | 0.612 | −13% |
| Tail, `ψ33`, `f` ∝ s, fast θ_E, no gyre | 16 | 0.485 | 0.363 | 21.5° | 0.618 / 32.0° | 0.203 | 0.491 | 0.323 | 0.664 | −12% |
| Tail, `ψ33`, `f` ∝ s, no gyre | 15 | 0.489 | 0.377 | 21.4° | 0.626 / 31.5° | 0.203 | 0.471 | 0.338 | 0.645 | −13% |
| Tail, `ψ33`, no gyre | 14 | 0.533 | 0.400 | 22.0° | 0.760 / 33.0° | 0.204 | 0.498 | 0.404 | 0.890 | +30% |

Tapering costs some accuracy at 18 numbers (speed 0.347 → 0.356, direction 19.4° → 20.9°), so some of the ladder's gain may have come from the polar flow absorbing other errors.

### Coefficients of the 22-number ladder function

This version still has the polar defect. Before the rescale of ×1.068 (applied to U_0 and G_0):

| Term | Coefficients |
|---|---|
| Speed | U_0 = 9.06, b_U = 0.216, c_U = −0.436 |
| Switch | P_t = 13.6 d |
| Core size | θ_E = 0.718, θ_W = 0.343, c_t = −0.377 |
| Fall-off | q = 6.87, dq = 11.1, b_s = 0.497 |
| `f` | f_slow = −0.0563, f_fast = 0.468 |
| Shift | λ0 = −5.08° |
| Jet | J_0 = 1.29, a_J = −0.684, c_J = −a_J, d_J = 2.2, φ_J = 70.4°, w = 15.7°, no cutoff |
| Gyre | G_0 = 2.7, φ_g = 42.2°, λ_g = 90.7°, σ = 38.2° |

## 9. Caveats

- **Held-out planets:** no function developed here has been scored on them. They are not fully unseen, however: the first per-planet comparison and the regime analysis used all planets, and the existing `simple` joint fit was scored on the held-out half.
- **Selection on VAL:** forms were compared repeatedly on VAL, so its scores are somewhat optimistic.
- **Thin benchmark at fast rotation:** only 11 same-climate planets at 4–8 d, and failures are concentrated at high P0 and fast rotation. This also makes the overall same-climate median easier to match.
- **Same-climate planets differ:** matching the climate under UM needs a substantially different planet (median CO2 change 0.7 dex, stellar mass 8%).
- **GCM disagreement:** UM and ExoCAM differ most at fast rotation, so features fitted there may be ExoCAM-specific. The west-shifted core is one candidate: the slow-rotator stagnation point is about 5° west in ExoCAM but 5° east in UM.
- **Nightside:** speed remains about 50% too low at 22 numbers, with direction error near 43°.
- **Greedy pruning:** the ladder removes one thing at a time, so a better combination may exist.
- **High-P0 `ustar3`:** the low bias above 4 bar was seen in earlier functions and was not re-examined for the latest form.

## 10. Status and reproducibility

- **Figures:** `figures/wind_*.png` and `figures/emulator_wind_*.png`. These are gitignored.
- **Code:** the fitting scripts (`wind_form4.py` and its predecessors) lived in a temporary session directory and have since been cleared, along with the generated data. They are not in the repository, and the scripts could be rebuilt from the session transcript.
- **Repository gaps:** `spherical_harmonic_winds` and `wind_decomposition_test` are imported by `converging_wind_test.py`, `joint_wind_fit.py` and `parametric_wind_test.py`, but are missing from the repository, so those scripts do not run from a fresh checkout.

## 11. Next steps

1. Refit the 22-number form with the tapered gyre and converged iterations, then compare it with the 14- and 16-number functions.
2. Choose one function and score it once on the held-out planets.
3. Move the chosen function and its fitting code into the repository.
4. If the gap needs closing further, focus on the 15–30 d transition and on how the wind pattern varies between planets at the same ΔT, P0 and P_rot.
