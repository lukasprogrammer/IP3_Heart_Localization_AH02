# IP3 Heart Sound Localization — coding context

## Project
- TU Delft EE2L1 IP3 "Heart Sound Localization" (manual: IP3_Heart_Sound_Manual_2026Nov10.pdf, in the claude.ai Project).
- Goal: localize the 4 heart valves with a 6-microphone stethoscope array (beamforming / MUSIC), then separate the sources.
- I work in the localization subgroup: **M3 (Ch. 6, basic direction finding)** and **M4 (Ch. 7, SVD + MUSIC)**, then system integration (Ch. 9 3D localization, Ch. 11 source separation, Ch. 12 integration).
- Theory is discussed in a separate claude.ai chat; this repo/session is for **coding and debugging**.

## How I want help
- Guide me step by step with hints first; explain *why* an error happens before giving the fix.
- Give the full code line/function when I explicitly ask for it.
- I am on Windows, VSCode, Python 3.12, numpy/scipy/matplotlib.

## Conventions (follow the manual)
- Data matrix `X` has shape (M, N): one microphone per row.
- Angles in radians internally; θ = 0 is broadside. Plot axes in degrees.
- Steering vector (Eq. 6.2), ULA: `a_m(θ) = exp(-j·2π·F0·m·(d/v)·sin θ)`, m = 0..M-1.
- Hermitian transpose: `a.conj().T`. Covariance `Rx = X @ X.conj().T / N`.
- Matched beamformer: `P_mbf(θ) = aᴴ Rx a`. MVDR: `P_mvdr(θ) = 1 / (aᴴ Rx⁻¹ a)` (use `np.linalg.solve`, not `inv`).
- Wideband: STFT (Hann, 50% overlap) → covariance per bin `Rx_k = X_k X_kᴴ / L` → average P_k(θ) over band [F1, F2]. Never take abs() of the STFT (direction is in the phase).
- Lab array: M = 6, d = 10 cm, v = 343 m/s, Fs = 48 kHz. Aliasing above v/(2d) ≈ 1715 Hz. Narrowband per bin needs N_w ≫ 1.44·Fs·D/v ≈ 100 (D = 0.5 m).
- Heart array: 2×3, 5 cm spacing, v_body ≈ 60 m/s.

## Required function signatures (from the manual)
- `a_lin(theta, M, d, v, F0)` — returns (M,) for scalar θ, or (M, Q) for Q angles.
- `matchedbeamformer(Rx, th_range, M, d, v, f0)`, `mvdr(Rx, th_range, M, d, v, f0)`
- `narrowband_Rx(X, fs, nperseg)` → `f_bins (K,)`, `Rx_all (K, M, M)`
- M4: `music(Rx, Q, M, th_range, d, v, f0)`
- Ch. 9: `a_z(s_position, mic_positions, M, v, f0)`, `music_z(...)`, `mvdr_z(...)`

## Current status (2026-11-20)
- Working on **assignment 6.2.6**: recreate Fig. 6.4 (|aᴴ(θ)a(θ0)|², M = 7, θ0 = 0°) for several Δ = d/λ, then vary θ0.
- Fixed so far:
  1. `a_lin` returned a scalar → must use `m = np.arange(M)` in the exponent, not `M`.
  2. Passed the whole angle array into `a_lin` inside the loop → pass the loop variable `th`.
- Working loop:
  ```python
  th_range = np.deg2rad(np.linspace(-90, 90, 721))
  a0 = a_lin(theta0, M, d, v, F0)
  P = np.zeros(len(th_range))
  for i, th in enumerate(th_range):
      a = a_lin(th, M, d, v, F0)
      P[i] = np.abs(a.conj() @ a0) ** 2
  ```
  Check: P at θ = 0° should equal M² = 49.
- Next: loop over Δ ∈ {0.25, 0.5, 1}, with d = Δ·v/F0, and explain resolution vs grating lobes (6.2.9-4); then 6.2.9 calculations; then 6.4 (matched BF + MVDR, two sources); then 6.6 experiments.

### Code state in repo (checked 2026-11-20, Claude Code session)
- Repo: `IP3_Heart_Localization_AH02`, branch `Direction`. Files: `Chapter_2_Assignment/Plots.py` (heartbeat plots, Ch. 2), `Chapter_2_Assignment/theta` (the 6.2.6 script, untracked).
- `theta` currently runs with F = 500 Hz, M = 7, Δ = 1 (single Δ, no loop yet). `a_lin` uses `m = np.arange(M)` ✔ and the loop passes the scalar `angle` ✔.
- Open issues spotted (to be worked through hint-first, not yet fixed):
  1. File has no `.py` extension → VSCode won't treat it as Python; rename (e.g. `m3_fig6_4.py`) and probably move out of `Chapter_2_Assignment/`.
  2. `plt.plot(theta, P)` plots radians but the label says degrees → plot `np.rad2deg(theta)`.
  3. Grid is `linspace(-90, 90, 180)` (non-integer step, misses 0° exactly) → use 721 points as in the working loop above.
  4. Leftover line `alin = a_lin(100, ...)` (100 rad, unused) → remove.
  5. `a_lin` only works for scalar θ; manual signature wants (M, Q) for Q angles → later use `np.outer(m, tau)` (needed for vectorised matched BF / MVDR in 6.4).
- `m3_direction_finding.py` reference is **not** in the repo yet.

### 6.4 Two sources — done (2026-11-20, written by Claude on request)
- File: `M3_Direction_Finding/m3_two_sources.py`, rewritten in the style of the user's own 6.2.6 script (flat script, `for i, angle in enumerate(theta)` loops, same variable names v/F/M/lam/delta/d). `a_lin` = user's version + an `else` branch giving (M, Q) via `m[:, None] * tau[None, :]` for a list of angles. Functions: `make_Rx(theta0, M, d, v, F0, SNR)` (manual's data model), `matchedbeamformer` and `mvdr` (loop per angle, `np.linalg.solve`), assignment 3 computed inline at the bottom.
- Settings: M = 7, Δ = 0.5, f0 = 500 Hz, SNR = 10 dB, sources at 0° and 15° (the angles in Fig. 6.7 couldn't be read from the PDF text; adjust if the figure differs).
- Results verified: MBF peaks are biased to 1.5° / 13.5° and nearly merged (resolution ≈ 180°/M ≈ 25°); MVDR peaks sit exactly at 0° / 15°. Assignment 3: |wᴴa(θ0)| = 1, |wᴴa(θ1)| = 0.0015 → MVDR nulls the other source.
- Single-source check: P_mbf(0°) = M² = 49 ✔.
- Python note: the interpreter with numpy/scipy/matplotlib on this PC is `py` (3.15, `C:\Users\natch\AppData\Local\Python\pythoncore-3.15-64`), not the msys `python` on PATH.
- User's own 6.4 version in `Chapter_2_Assignment/theta` — **fixed by Claude on request (2026-11-22), runs correctly** (P(0°) = 49, MBF peaks 1.5°/13.5°, MVDR peaks 0°/15°). Fixes made:
  1. `a_lin`: user had `np.sin(th)[None, :]` → IndexError for a scalar (0-d array). Now `np.atleast_1d(theta)` → always (M, Q), then `[:, 0]` back to (M,) for scalar input. Uses `F0` not global `F`.
  2. `give_Rx(theta0, ...)` instead of all 721 scan angles.
  3. `matchedbeamformer`: `aᴴ Rx a` (user had `aᴴ A0 * Rx`), local `Pmbf`, `np.real`.
  4. Added `mvdr`, second subplot and `plt.show()`; removed unused imports and `.item()`.
- File still has no `.py` extension (open issue 1 above).
- Still open from 6.2: Δ loop for Fig. 6.4, 6.2.9 calculations, fix issues 1–4 in `Chapter_2_Assignment/theta`. Next after that: 6.5/6.6 (STFT `narrowband_Rx`, wideband averaging, experiments).

## Reference
`m3_direction_finding.py` (reference implementation of a_lin, matched BF, MVDR, narrowband_Rx, band averaging, with self-test) — use to check my code, don't just paste it in.
