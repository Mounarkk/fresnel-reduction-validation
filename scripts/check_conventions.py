"""
Do our colour conventions match Mitsuba's internals?  Two silent mismatches
would contaminate every Delta-E reported by compare.py:
  1. CMF tables: results/cmf.npz must agree with the CIE 1931 table the
     spectral film uses (mi.cie1931_xyz).  compute_cmf.py samples that table
     directly; this script proves the stored S still matches it.
  2. XYZ<->sRGB matrix: the IEC 61966-2-1 matrix hardcoded in the RGB path
     must equal mi.xyz_to_srgb.

Both comparisons are up to a global scale (Phi is invariant to a common
rescaling of S).  Exits non-zero on mismatch.
"""
import numpy as np
import mitsuba as mi

from _paths import RESULTS

mi.set_variant('scalar_spectral')

TOL = 1e-3   # relative, after normalization
ok = True

# ── 1. CMF tables ─────────────────────────────────────────────────────────────
wl_nm = np.load(RESULTS / 'spectral_coeffs.npz')['wl_nm']
S_ours = np.load(RESULTS / 'cmf.npz')['S']                     # (N, 3), common-ybar normalized

S_mi = np.array([mi.cie1931_xyz(float(wl)) for wl in wl_nm]) # (N, 3), unnormalized
S_mi = S_mi / S_mi[:, 1].sum()                               # same normalization as ours

err = np.abs(S_ours - S_mi)
rel = err.max() / S_mi.max()
print(f"CMF tables   — max abs diff: {err.max():.3e}   max rel diff: {rel:.3e}", end='')
if rel < TOL:
    print("   OK")
else:
    ok = False
    print("   MISMATCH ✗")
    for k, name in enumerate('xyz'):
        j = err[:, k].argmax()
        print(f"    worst {name}-bar at {wl_nm[j]:.0f} nm: ours={S_ours[j,k]:.6f} mitsuba={S_mi[j,k]:.6f}")

# ── 2. XYZ -> sRGB matrix ─────────────────────────────────────────────────────
XYZ_TO_sRGB_ours = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
# Extract Mitsuba's matrix column-by-column from unit XYZ vectors.
M_mi = np.column_stack([np.array(mi.xyz_to_srgb(mi.Color3f(*col))).ravel()
                        for col in np.eye(3)])

err_m = np.abs(XYZ_TO_sRGB_ours - M_mi).max()
print(f"XYZ->sRGB    — max abs diff: {err_m:.3e}", end='')
if err_m < TOL:
    print("   OK")
else:
    ok = False
    print("   MISMATCH ✗")
    print("  ours:\n", XYZ_TO_sRGB_ours, "\n  mitsuba:\n", M_mi)

print("\nAll conventions match Mitsuba." if ok else "\nCONVENTION MISMATCH — fix before trusting any Delta-E.")
raise SystemExit(0 if ok else 1)
