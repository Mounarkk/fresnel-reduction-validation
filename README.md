# Reduced conductor Fresnel: validation pipeline

Python / Mitsuba 3 validation of a **spectrally reduced, real-time-capable
Fresnel model for conductors**.  The non-orthogonal spectral reduction of
Fichet, Belcour & Barla (2024) only covers angle-independent operators; this
work extends it to the angle-dependent specular Fresnel of a metal by
factoring the Fresnel operator on the four universal angular bases of
Belcour, Bati & Barla (2020).  The result is a 3x3 matrix

    M_F(theta) = sum_i b_i(theta) * Phi_i          (Phi_i precomputed, 3x3 each)

applied to transported radiance: one basis fetch, four matrix multiply-adds
and one matrix-vector product per pixel.

All results are checked against full spectral renders.  Clone the repository,
create the environment, and every figure, number and check regenerates from
`scripts/`.  `doc/main.tex` covers the theory, the experiments, the analysis
of the residual tint and the limitations.  Usage is described below.

## Layout

```
scripts/
  load_ior.py              gold eta(lambda), kappa(lambda) -> results/gold_ior.npz
  extract_basis.py         b0..b3 from the authors' table   -> results/basis_functions.npz
  compute_coeffs.py        per-wavelength c0..c3            -> results/spectral_coeffs.npz
  compute_cmf.py           CMF basis S and dual Sb           -> results/cmf.npz
  compute_phi.py           Phi_i = S^T diag(c_i) Sb         -> results/Phi.npz
  compute_split_sum.py     b_bar_i(NdotV, alpha) LUT         -> results/split_sum.npy
  check_conventions.py     check: S and XYZ->sRGB match Mitsuba's internals
  export_godot.py          check + export: Psi_i and basis  -> results/godot_gold.json
  audit_tint.py            offline dissection of the tint   -> results/audit_tint.txt
  make_illuminants.py      achromatic HDRI + D65 / FL1 SPDs -> scene/
  render_reference.py      full spectral ground truth (Mitsuba)
  render_ours.py           spectral render, 4-basis Fresnel only
  render_ours_direct.py    RGB render, reduced 3x3 matrix in a custom integrator
  compare.py               RMSE + Delta-E 2000 + figure     -> results/comparison*.png
  fresnel_tools.py         exact conductor Fresnel (from the authors' helper module)
  _paths.py                repository-relative paths
data/
  gold_refractive_index.csv        complex IOR of gold, wavelength in micrometres
  belcour2020/fresnel_curves.hpp   the universal basis table, from the authors' Mitsuba plugin
scene/                     Mitsuba XML scenes, spectral IOR and illuminant files, uffizi.hdr
results/                   committed figures and small artefacts (regenerable)
doc/                       LaTeX companion document (Overleaf-ready)
```

## Quick start

```sh
make venv            # python3 -m venv .venv + pinned requirements (Mitsuba 3 included)
make precompute      # seconds: IOR -> basis -> coefficients -> CMF -> Phi
make check           # convention check + Godot export with its two checks, exit 1 on failure
make audit           # offline tint dissection -> results/audit_tint.txt
make illuminants     # scene/uffizi_gray.exr and the two SPD files
make render          # the 12 Mitsuba renders (4 illuminants x 3 paths) -- long
make compare         # the 4 comparison figures and their Delta-E numbers
make splitsum        # the split-sum LUT (Monte Carlo, all cores) -- minutes
make all             # everything above, in dependency order
```

Without `make`: `.venv/bin/python scripts/<name>.py` from any directory.  The
render scripts take `--scene`, `--out` and `--param k=v`; see the `render`
target in the Makefile for the twelve invocations.

Requires Python 3.12 and the pinned versions in `requirements.txt`.  Mitsuba
runs on the CPU (`llvm_*` variants); no GPU is needed.

## What each stage establishes

| Stage | Question it answers | Check |
|---|---|---|
| `compute_coeffs.py` | Does the 4-basis decomposition reproduce gold's exact Fresnel curve? | max abs error at 550 nm < 0.01 (observed 0.0036) |
| `compute_cmf.py` | Is the dual basis a true dual? | `S^T Sb == I` to 1e-6 |
| `check_conventions.py` | Do our CMF table and sRGB matrix equal Mitsuba's? | exact / 8.6e-5 |
| `export_godot.py` | Does the exported data assemble the right operator? | pre-fold round-trip 3e-16; 4-basis vs exact reduced operator 2.2e-3 |
| `render_ours.py` vs reference | Is the angular decomposition accurate in a renderer? | Delta-E 2000 mean 0.02-0.04 |
| `render_ours_direct.py` vs reference | Is the reduced matrix accurate in a renderer? | Delta-E 2000 mean 0.8-2.7, depends on the illuminant (see below) |
| `audit_tint.py` | Where does the reduced path's residual come from? | reproduces the rendered tint in closed form; metameric black |

## Results

Gold sphere, GGX alpha = 0.05, 2048 spp, 1024 x 1024, `direct` integrator.
Mean CIEDE2000 against the full spectral reference:

| Illuminant | Spectral path: Delta-E / RMSE | Reduced-matrix path: Delta-E / RMSE |
|---|---|---|
| Uffizi HDRI (RGB, uplifted) | 0.020 / 0.0008 | 2.11 / 0.030 |
| Uffizi, achromatic | 0.020 / 0.0008 | 2.04 / 0.029 |
| D65, natively spectral | 0.044 / 0.0009 | 2.70 / 0.032 |
| FL1, natively spectral | 0.041 / 0.0008 | 0.78 / 0.011 |

(RMSE on linear HDR radiance.  Every number regenerated from this repository
and identical to the original runs to three decimals.)

What the figures show:

* **The angular 4-basis decomposition is accurate.**  The spectral
  path is at the noise floor under every illuminant.
* **The reduced-matrix path has a systematic tint** (about -6% red, +6%
  green for gold under D65).  It comes from the dual-basis reduction rather
  than from the renderer: `audit_tint.py` reproduces the rendered pixel values
  to three decimals with no Monte Carlo involved.  The error is larger under
  smooth illuminants than under the spiky FL1 because a smooth spectrum has
  more energy outside the span of the colour-matching functions (55% of
  D65's, in L2), and gold's wavelength-varying Fresnel brings that part back
  into the visible.

`results/audit_tint.txt` also compares alternatives.  A reconstruction
operator anchored on smooth daylight removes the error on every daylight
illuminant at the same runtime cost, and over several bounces the matrix form
keeps that anchoring stable while a diagonal, per-channel prefilter degrades
(0 -> 4.8 -> 9.8 Delta-E over three bounces).  See doc sections 7 and 8.

## Reading the figures

`results/comparison*.png`: top row, reference | spectral path | reduced path;
bottom row, per-pixel Delta-E 2000 maps for the two paths on a shared scale
capped at 5.  In the Uffizi comparison the error map is non-zero on the
background, where no material is involved: the reference upsamples the RGB
environment map to a spectrum and the RGB path does not, and the two metamers
disagree.  The three other illuminants were added to remove this effect.
Black speckles in the reference are pixels the spectral integrator produced
as NaN; they are zeroed before comparison and appear as isolated bright dots
in the error maps.  `results/basis_plot.png` and
`results/coeffs_plot.png` show the four bases and the 550 nm fit;
`results/split_sum_plot.png` the four channels of the preintegrated LUT.

## Notes

* The colour-matching basis is normalised by a single common factor (the
  y-bar sum) so that it matches Mitsuba's spectral film.  The sibling
  fluorescence repository normalises per column; the two `S` matrices are
  not interchangeable.
* The basis functions are read from the authors' published table; they are
  not re-derived by SVD here.
* The split-sum LUT is computed and plotted but no render uses it: all
  renders use the exact-angle basis at alpha = 0.05, the mirror case chosen
  for the first engine scene.  The rough path is validated only through the
  LUT's own sanity checks.
* Only one material (gold) and one geometry are covered.
* `.exr` renders and `scene/uffizi_gray.exr` are not committed (regenerable,
  ~140 MB); `make compare` therefore needs `make render` first.

## Related repositories

* `spectral-reduction-validation` -- the reflectance / fluorescence half of
  the same reduction framework (angle-independent operators).
* `pseudo-spectral-godot` -- the Godot 4.6 project; `addons/fresnel_conductor/`
  consumes `results/godot_gold.json`.

## References

* A. Fichet, L. Belcour, P. Barla. *Non-Orthogonal Reduction for Rendering
  Fluorescent Materials in Non-Spectral Engines.* CGF 2024.
* L. Belcour, M. Bati, P. Barla. *Bringing an Accurate Fresnel to Real-Time
  Rendering: a Preintegrable Decomposition.* SIGGRAPH Talks 2020.
* W. Jakob, J. Hanika. *A Low-Dimensional Function Space for Efficient
  Spectral Upsampling.* CGF (EG) 2019.
* B. Karis. *Real Shading in Unreal Engine 4.* SIGGRAPH Courses 2013.
* E. Heitz. *Sampling the GGX Distribution of Visible Normals.* JCGT 2018.
* G. Wyszecki, W. S. Stiles. *Color Science*, 2nd ed. (metameric blacks).
* I. Mallett, C. Yuksel. *Spectral Primary Decomposition for Rendering with
  sRGB Reflectance.* EGSR 2019.
