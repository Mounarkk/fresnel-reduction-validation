"""
Export the reduced conductor Fresnel to a Godot-loadable form.

Consumes the two precomputed artifacts of this pipeline:
  * results/Phi.npz              - Phi0..Phi3, four 3x3 XYZ-space reduction matrices.
  * results/basis_functions.npz  - b0..b3, the universal Belcour basis on a uniform
                                   cos(theta) grid of 1000 samples (b0 == 1).

and writes a single self-contained JSON (results/godot_gold.json) holding:
  * "psi"   : four 3x3 matrices Ψ_i = M_{rgb<-xyz} · Φ_i · M_{xyz<-rgb}, ROW-MAJOR.
              These are pre-folded into linear sRGB (exactly render_ours_direct.py's
              PSI), so the Godot shader — whose radiance is linear sRGB — can apply
              M_F(cosθ) = Σ_i b_i(cosθ) · Ψ_i  directly, no colour conversion at runtime.
  * "basis" : b0..b3 as four length-1000 arrays on the uniform cosθ∈[0,1] grid.
              The Godot side packs these into a 1000×1 RGBA32F texture (float, so the
              NEGATIVE b3 values survive) and samples it at U = cosθ.

Runtime cost target (unchanged from the note): 1 RGBA fetch + 4 mat3 MADs + 1 mat3×vec3.

The reduction data is illuminant-free and material-specific (gold here). Swapping the
metal = regenerate Phi.npz upstream and re-run this. Nothing here depends on the env,
roughness, or the scene.

Before writing, checks that the sRGB pre-fold inverts back to Phi and that the
4-basis assembly  sum_i b_i Phi_i  agrees with the exact reduced operator
S^T diag(F(theta)) Sb  recomputed from the gold IOR at five angles.
"""
import json

import numpy as np

import fresnel_tools as ft
from _paths import RESULTS

# ── Colour-space fold (identical to render_ours_direct.py) ────────────────────
XYZ_TO_sRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
sRGB_TO_XYZ = np.linalg.inv(XYZ_TO_sRGB)

phi = np.load(RESULTS / 'Phi.npz')
PSI = [XYZ_TO_sRGB @ phi[f'Phi{i}'] @ sRGB_TO_XYZ for i in range(4)]   # linear-sRGB space

basis = np.load(RESULTS / 'basis_functions.npz')
b = [basis[f'b{i}'].astype(np.float64) for i in range(4)]              # each length 1000
cT = basis['cT'].astype(np.float64)
N = len(cT)
assert all(len(bi) == N for bi in b)
assert np.allclose(np.diff(cT), np.diff(cT)[0])
assert np.allclose(b[0], 1.0)

# -- Check (a): the sRGB pre-fold must invert back to Phi ----------------------
err_fold = max(np.abs(sRGB_TO_XYZ @ PSI[i] @ XYZ_TO_sRGB - phi[f'Phi{i}']).max() for i in range(4))
print(f"sRGB pre-fold round-trip vs Phi: max|diff| = {err_fold:.2e}")

# -- Check (b): 4-basis assembly vs the exact reduced operator ------------------
# The shader assembles M_F(cos t) = sum_i b_i(cos t) Phi_i (in XYZ, before the fold).
# Independently, the exact reduced operator at that angle is S^T diag(F(cos t)) Sb
# with F the true conductor Fresnel of gold.  They differ only by the 4-basis fit.
ior = np.load(RESULTS / 'gold_ior.npz'); cmf = np.load(RESULTS / 'cmf.npz')
S, Sb, eta, kap = cmf['S'], cmf['Sb'], ior['eta'], ior['kappa']
err_fit = 0.0
for ct in (1.0, 0.7071, 0.5, 0.2588, 0.1201):
    k = int(round(ct * (N - 1)))                             # same rounding as the renderer
    M4 = sum(b[i][k] * phi[f'Phi{i}'].astype(np.float64) for i in range(4))
    F  = np.array([ft.fresnel(ct, eta[j], kap[j]) for j in range(len(eta))])
    Mx = (S.T * F) @ Sb
    err_fit = max(err_fit, np.abs(M4 - Mx).max())
print(f"4-basis M_F vs exact reduced operator: max|diff| = {err_fit:.2e} over 5 angles")
assert err_fold < 1e-5
assert err_fit < 0.02

# Sanity: at cosθ=1 (normal incidence) M_F should be ~diag(F0) in XYZ; check white-anchor
# behaviour indirectly by reporting the normal-incidence sRGB gold matrix.
M_normal = sum(b[i][-1] * PSI[i] for i in range(4))
print("M_F(cosθ=1) [linear sRGB] =\n", np.round(M_normal, 4))
print("  → column-sums (energy per input channel):", np.round(M_normal.sum(0), 4))

# ── Write the JSON ────────────────────────────────────────────────────────────
out = {
    "material": "gold",
    "colour_space": "linear_sRGB",
    "note": ("psi[i] is Ψ_i = M_rgb<-xyz · Φ_i · M_xyz<-rgb, ROW-MAJOR (out[r]=Σ_c psi[r][c]·L[c]). "
             "M_F(cosθ)=Σ_i b[i]·Ψ_i. basis[i] sampled on uniform cosθ∈[0,1], 1000 pts, index U=cosθ. "
             "b0≡1. b3 has NEGATIVE values → pack basis into a FLOAT (RGBA32F) texture, not 8-bit."),
    "grid_size": N,
    "psi": [PSI[i].tolist() for i in range(4)],           # 4 × 3×3 row-major
    "basis": [b[i].tolist() for i in range(4)],           # 4 × 1000
}
out_path = RESULTS / 'godot_gold.json'
with open(out_path, 'w') as f:
    json.dump(out, f)
print(f"Saved results/godot_gold.json ({out_path.stat().st_size/1024:.1f} KB)")
print("b3 range (why float texture): [%.4f, %.4f]" % (b[3].min(), b[3].max()))
