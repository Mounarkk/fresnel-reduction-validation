"""
Dissect the residual tint of the reduced-matrix path OFFLINE, in pure linear
algebra.  No renderer, no Monte Carlo, no tonemapping: if the tint shows up
here, it is a property of the mathematics, not of Mitsuba or the integrator.

Question under audit: ours_direct desaturates gold (~ -6% R, +6% G at normal
incidence under D65). Candidate causes and candidate fixes are all evaluated
against the same exact spectral computation:

  exact:      x = S^T ( F(theta,.) * E )                (ground truth XYZ)
  reduced:    x = Phi(theta) (S^T E),  Phi = S^T diag(F) R,  R = reconstruction
  where R is the design choice:
    [A]  R = Sb = S(S^T S)^{-1}          (Fichet dual basis — what we do now)
    [B]  R = W S (S^T W S)^{-1}, W=diag(E_prior)   (illuminant-weighted dual)
    [C]  R = L (S^T L)^{-1}, L = CIE daylight S0,S1,S2  (exact on daylights)
    [D]  R anchored on {D65, FL1 (or FL2), A}           (exact on those three)
  baselines (diagonal, what engines do):
    [E]  rho_k = (S^T F)_k / (S^T 1)_k        flat-weighted diagonal
    [F]  rho_k = (S^T F.E)_k / (S^T E)_k      Ward-prefiltered (exact @ E, 1 bounce)
    [G]  Belcour-RGB: eta,kappa CMF-averaged per channel, then Fresnel
  and the sanity checks:
    - 4-basis M_F vs exact-operator Phi(theta): isolates Belcour-fit error.
    - XYZ transport vs sRGB-folded transport (Psi): must be identical (linear).
    - metameric-black decomposition: error == S^T diag(F) (I - Sb S^T) E.

Angles: cos(theta) in {1.0, 0.7071, 0.5, 0.2588, 0.1201}.
Illuminants: E (flat), D65, FL1.
Metric: Delta-E 2000 on Lab, mirroring compare.py's convention
(XYZ -> linear sRGB -> Lab with skimage-style D65 white), applied to the
un-tonemapped colour (values are O(1), comparable to render pixel values).

"""
import numpy as np
import colour as colour_lib
import skimage.color

import fresnel_tools as ft
from _paths import RESULTS


# ── Data ──────────────────────────────────────────────────────────────────────
ior = np.load(RESULTS / 'gold_ior.npz')
wl, eta, kap = ior['wl_nm'], ior['eta'], ior['kappa']
cmf = np.load(RESULTS / 'cmf.npz')
S, Sb = cmf['S'], cmf['Sb']                       # (N,3), Mitsuba's own CMF table
bas = np.load(RESULTS / 'basis_functions.npz')
b = [bas[f'b{i}'] for i in range(4)]
cT_grid = bas['cT']
co = np.load(RESULTS / 'spectral_coeffs.npz')
c_i = [co[f'c{i}'] for i in range(4)]             # spectral coefficient curves

XYZ_TO_sRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])

def illum(name):
    if name == 'E(flat)':
        v = np.ones_like(wl)
    else:
        sd = colour_lib.SDS_ILLUMINANTS[name]
        v = np.interp(wl, sd.wavelengths, sd.values, left=0.0, right=0.0)
    # same normalization as the renders: film luminance Y = 1
    return v / (S.T @ v)[1]

def F_of(ct):
    return np.array([ft.fresnel(ct, eta[j], kap[j]) for j in range(len(wl))])

def to_lab(xyz):
    rgb = XYZ_TO_sRGB @ xyz
    return skimage.color.rgb2lab(np.clip(rgb, 0, None)[None, None, :].astype(np.float64))[0, 0]

def dE(x_test, x_exact):
    return float(colour_lib.difference.delta_E_CIE2000(to_lab(x_exact), to_lab(x_test)))

# ── Reconstruction operators R (N x 3) ────────────────────────────────────────
def R_weighted(E_prior):
    W = E_prior
    SW = S * W[:, None]
    return SW @ np.linalg.inv(S.T @ SW)

def R_anchored(spectra):
    L = np.column_stack(spectra)                  # (N,3)
    return L @ np.linalg.inv(S.T @ L)

R_ops = {'[A] dual basis (ours)': Sb}

d65_full = illum('D65')
R_ops['[B] D65-weighted dual'] = R_weighted(d65_full)

try:  # CIE daylight components S0,S1,S2
    dl = colour_lib.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES
    S012 = [np.interp(wl, dl[k].wavelengths, dl[k].values, left=0.0, right=0.0)
            for k in ('S0', 'S1', 'S2')]
    R_ops['[C] daylight S0S1S2'] = R_anchored(S012)
except Exception as e:
    print(f"(skipping [C]: {e})")

R_ops['[D] anchored D65/FL1/A'] = R_anchored([illum('D65'), illum('FL1'), illum('A')])

# Belcour-RGB channel-averaged IOR (weights = CMFs, per channel)
w_ch = S / S.sum(axis=0, keepdims=True)           # (N,3) column-normalized
eta_ch = w_ch.T @ eta
kap_ch = w_ch.T @ kap

# ── Main loop ─────────────────────────────────────────────────────────────────
angles = [1.0, 0.7071, 0.5, 0.2588, 0.1201]
illums = ['E(flat)', 'D65', 'FL1']

methods = list(R_ops.keys()) + ['[A4] 4-basis M_F', '[E] diag flat', '[F] diag Ward@E', '[G] Belcour-RGB']
results = {m: {il: [] for il in illums} for m in methods}

for il in illums:
    E = illum(il)
    cin = S.T @ E
    for ct in angles:
        F = F_of(ct)
        x_exact = S.T @ (F * E)

        for name, R in R_ops.items():
            Phi = (S.T * F) @ R
            results[name][il].append(dE(Phi @ cin, x_exact))

        # 4-basis version of [A]: M_F = sum_i b_i(ct) Phi_i (exact-angle path)
        j = int(round(ct * 999))
        M_F = sum(b[i][j] * ((S.T * c_i[i]) @ Sb) for i in range(4))
        results['[A4] 4-basis M_F'][il].append(dE(M_F @ cin, x_exact))

        # diagonal baselines
        rho_flat = (S.T @ F) / (S.T @ np.ones_like(F))
        results['[E] diag flat'][il].append(dE(rho_flat * cin, x_exact))
        rho_ward = (S.T @ (F * E)) / cin
        results['[F] diag Ward@E'][il].append(dE(rho_ward * cin, x_exact))
        F_ch = np.array([ft.fresnel(ct, eta_ch[k], kap_ch[k]) for k in range(3)])
        results['[G] Belcour-RGB'][il].append(dE(F_ch * cin, x_exact))

# ── Report ────────────────────────────────────────────────────────────────────
hdr = "cosT:      " + "".join(f"{ct:>8.3f}" for ct in angles) + "     mean"
for il in illums:
    print(f"\n═══ illuminant {il} — ΔE00 vs exact spectral ═══")
    print(hdr)
    for m in methods:
        v = results[m][il]
        print(f"{m:<24}" + "".join(f"{x:8.3f}" for x in v) + f" {np.mean(v):8.3f}")

# ── Cross-checks ──────────────────────────────────────────────────────────────
print("\n═══ cross-checks ═══")
E = illum('D65'); cin = S.T @ E; F0 = F_of(1.0)
x_exact = S.T @ (F0 * E)
Phi = (S.T * F0) @ Sb
print(f"render link @cosT=1, D65: exact sRGB  = {np.round(XYZ_TO_sRGB @ x_exact, 4)}")
print(f"                          reduced sRGB= {np.round(XYZ_TO_sRGB @ (Phi @ cin), 4)}")
print( "                          (compare: reference_d65 px [1.0226 0.7199 0.3412],"
       " ours_direct_d65 px [0.9563 0.7626 0.3272])")

# XYZ transport vs sRGB-folded transport must agree exactly (pure linearity)
M = XYZ_TO_sRGB; Minv = np.linalg.inv(M)
Psi = M @ Phi @ Minv
diff = np.abs(M @ (Phi @ cin) - Psi @ (M @ cin)).max()
print(f"XYZ vs sRGB-folded single-bounce max abs diff: {diff:.2e}  (must be ~0)")

# metameric black identity: error = S^T diag(F) (I - Sb S^T) E
err_direct = S.T @ (F0 * E) - Phi @ cin
black = E - Sb @ (S.T @ E)
err_via_black = S.T @ (F0 * black)
print(f"metameric-black identity residual: {np.abs(err_direct - err_via_black).max():.2e}  (must be ~0)")
print(f"black energy fraction ||B||/||E||: {np.linalg.norm(black)/np.linalg.norm(E):.4f}")

# ── Extension 1: generalization to unseen illuminants ─────────────────────────
print("\n═══ generalization: mean ΔE00 over angles, per illuminant (incl. unseen) ═══")
gen_illums = ['E(flat)', 'A', 'D50', 'D65', 'D75', 'FL1', 'FL2', 'FL4']

# PCA-of-illuminants reconstruction: top-3 subspace of a plausible light set
train = np.column_stack([illum(n) for n in ['A', 'D50', 'D55', 'D65', 'D75',
                                            'FL1', 'FL2', 'FL3', 'FL4', 'E(flat)']])
U, sv, _ = np.linalg.svd(train, full_matrices=False)
R_pca = U[:, :3] @ np.linalg.inv(S.T @ U[:, :3])
R_ops2 = dict(R_ops); R_ops2['[H] PCA(10 illums)'] = R_pca

methods2 = list(R_ops2.keys()) + ['[E] diag flat', '[G] Belcour-RGB']
print(f"{'method':<24}" + "".join(f"{il:>9}" for il in gen_illums))
for m in methods2:
    row = []
    for il in gen_illums:
        E = illum(il); cin = S.T @ E
        errs = []
        for ct in angles:
            F = F_of(ct)
            x_exact = S.T @ (F * E)
            if m == '[E] diag flat':
                rho = (S.T @ F) / (S.T @ np.ones_like(F)); x = rho * cin
            elif m == '[G] Belcour-RGB':
                F_ch = np.array([ft.fresnel(ct, eta_ch[k], kap_ch[k]) for k in range(3)])
                x = F_ch * cin
            else:
                x = ((S.T * F) @ R_ops2[m]) @ cin
            errs.append(dE(x, x_exact))
        row.append(np.mean(errs))
    print(f"{m:<24}" + "".join(f"{v:9.3f}" for v in row))

# ── Extension 2: multi-bounce (does the matrix formalism pay off in depth?) ──
print("\n═══ two- and three-bounce error (cosT=0.7071 each bounce, ΔE00) ═══")
ct = 0.7071; F = F_of(ct)
print(f"{'method':<24}{'illum':<9}{'1-bounce':>9}{'2-bounce':>9}{'3-bounce':>9}")
for il in ['D65', 'FL1']:
    E = illum(il); cin = S.T @ E
    for m in list(R_ops2.keys()) + ['[F] diag Ward@E', '[E] diag flat']:
        row = []
        for n in (1, 2, 3):
            x_exact = S.T @ (F**n * E)
            if m == '[F] diag Ward@E':
                rho = (S.T @ (F * E)) / cin; x = rho**n * cin
            elif m == '[E] diag flat':
                rho = (S.T @ F) / (S.T @ np.ones_like(F)); x = rho**n * cin
            else:
                Phi = (S.T * F) @ R_ops2[m]
                x = np.linalg.matrix_power(Phi, n) @ cin
            row.append(dE(x, x_exact))
        print(f"{m:<24}{il:<9}" + "".join(f"{v:9.3f}" for v in row))

# ── Extension 3: Mallett-Yuksel sRGB primaries as R + conditioning ────────────
print("\n═══ Mallett & Yuksel 2019 sRGB primaries as reconstruction + conditioning ═══")
try:
    my = colour_lib.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
    B_my = np.column_stack([np.interp(wl, my.wavelengths, my.values[:, k],
                                      left=0.0, right=0.0) for k in range(3)])
    R_my = B_my @ np.linalg.inv(S.T @ B_my)
    R_ops3 = {'[A] dual basis (ours)': Sb, '[C] daylight S0S1S2': R_ops['[C] daylight S0S1S2'],
              '[I] Mallett-Yuksel sRGB': R_my}
    for m, R in R_ops3.items():
        print(f"{m:<26} cond(S^T R-pre)={np.linalg.cond(S.T @ (R * 1.0)):8.2f}", end='  ')
        row = []
        for il in gen_illums:
            E = illum(il); cin = S.T @ E
            errs = [dE(((S.T * F_of(ct)) @ R) @ cin, S.T @ (F_of(ct) * E)) for ct in angles]
            row.append(np.mean(errs))
        print(" | " + "".join(f"{v:8.3f}" for v in row) + f"   ({', '.join(gen_illums)})")
    # multi-bounce for Mallett
    ct = 0.7071; F = F_of(ct)
    for il in ['D65', 'FL1']:
        E = illum(il); cin = S.T @ E
        Phi = (S.T * F) @ R_my
        v = [dE(np.linalg.matrix_power(Phi, n) @ cin, S.T @ (F**n * E)) for n in (1, 2, 3)]
        print(f"[I] Mallett multi-bounce {il}: " + "".join(f"{x:8.3f}" for x in v))
except Exception as e:
    print(f"(Mallett-Yuksel unavailable: {e})")

# conditioning of the anchored variants (explains multi-bounce blowup)
for name in ('[C] daylight S0S1S2', '[D] anchored D65/FL1/A', '[H] PCA(10 illums)'):
    R = R_ops2.get(name) if name in R_ops2 else R_ops.get(name)
    print(f"cond(R) {name:<26}: {np.linalg.cond(R):10.2f}")
