"""
Build the CIE 1931 XYZ colour-matching basis  S  (N x 3) and its dual
Sb = S (S^T S)^-1  on the pipeline's wavelength grid.

S is sampled from Mitsuba's own internal CMF table and normalised by a
SINGLE common factor (the y-bar sum), so that it matches, by construction,
the spectral-to-XYZ conversion the reference renders use.  See the notes in
the code for why per-column normalisation would be wrong here.

Output: results/cmf.npz  (S, Sb)
"""
import numpy as np
import mitsuba as mi

from _paths import RESULTS

mi.set_variant('scalar_spectral')

def main():
    # Load wavelength grid from spectral coefficients (or recreate it)
    data = np.load(RESULTS / 'spectral_coeffs.npz')
    wl_nm = data['wl_nm']
    N_wl = len(wl_nm)

    # CIE 1931 XYZ CMFs, sampled from MITSUBA'S OWN internal table so that our
    # S (and therefore every Phi matrix) matches the reference film's
    # spectral->XYZ conversion by construction. Using an external CIE source
    # (colour-science 1nm data, the previous behaviour) disagrees with
    # Mitsuba's interpolated table by up to ~3% pointwise in low-amplitude
    # regions — a silent contributor to reference-vs-ours tint.
    S = np.array([mi.cie1931_xyz(float(wl)) for wl in wl_nm])  # (N, 3)
        
    # Standard CIE / Mitsuba luminance normalization: a SINGLE common factor
    # (1 / sum of the y-bar column) applied equally to all three channels, so
    # that y-bar integrates to 1 and X,Z keep their true relative CMF areas.
    # NOTE: normalizing each column by its own sum (the previous behaviour)
    # applies three *different* gains -> a fixed per-channel tint that does not
    # match Mitsuba's spectral->XYZ conversion in the reference render.
    S /= S[:, 1].sum()
    
    # Compute dual basis Sb = S @ inv(S.T @ S)
    Sb = S @ np.linalg.inv(S.T @ S)
    
    # Assert S.T @ Sb == I_3 (atol accounts for Mitsuba's float32 CMF table;
    # Phi is stored float32 downstream anyway)
    assert np.allclose(S.T @ Sb, np.eye(3), atol=1e-6), "Dual basis property failed S.T @ Sb != I_3"
    print("Validation passed: S.T @ Sb == I_3")
    
    # Save S and Sb
    out_path = RESULTS / 'cmf.npz'
    np.savez(out_path, S=S, Sb=Sb)
    print(f"Saved S and Sb to {out_path}")

if __name__ == '__main__':
    main()
