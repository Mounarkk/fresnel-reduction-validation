"""
Decompose the exact conductor Fresnel curve of gold, wavelength by
wavelength, onto the four universal basis functions:

    F(cos theta, lambda) ~= sum_i c_i(lambda) b_i(cos theta)

c_0 = F(1) and c_1 = 1 - c_0 are analytic; c_2, c_3 come from a 2x2 system
evaluated at the two probe angles (Cramer's rule), exactly as in the
authors' fresnelToCoeffs().

Output: results/spectral_coeffs.npz (c0..c3 on the 401-point grid),
        results/coeffs_plot.png (fit vs exact at 550 nm)
"""
import numpy as np
import matplotlib.pyplot as plt

import fresnel_tools as ft
from _paths import RESULTS

def main():
    print("Computing spectral coefficients...")
    
    # Load basis
    basis_data = np.load(RESULTS / 'basis_functions.npz')
    b0 = basis_data['b0']
    b1 = basis_data['b1']
    b2 = basis_data['b2']
    b3 = basis_data['b3']
    cT = basis_data['cT']
    i1 = basis_data['probe_i1']
    ct1 = basis_data['probe_ct1']
    i2 = basis_data['probe_i2']
    ct2 = basis_data['probe_ct2']

    # Load IOR
    ior_data = np.load(RESULTS / 'gold_ior.npz')
    wl_nm = ior_data['wl_nm']
    eta = ior_data['eta']
    kappa = ior_data['kappa']

    N_wl = len(wl_nm)
    c = np.zeros((4, N_wl))

    # Cramer's rule coefficients
    A11 = b3[i2]
    A12 = -b2[i2]
    A21 = -b3[i1]
    A22 = b2[i1]
    det = A11 * A22 - A12 * A21

    for j in range(N_wl):
        eta_j, kap_j = eta[j], kappa[j]

        # c0, c1 are analytic
        c0 = ft.fresnel(1.0, eta_j, kap_j)
        c1 = max(0.0, 1.0 - c0)

        # Residual at the two probe angles
        dF1 = ft.fresnel(ct1, eta_j, kap_j) - (c0 + c1 * b1[i1])
        dF2 = ft.fresnel(ct2, eta_j, kap_j) - (c0 + c1 * b1[i2])

        # Cramer's rule on the 2x2 system
        c2 = (A11 * dF1 + A21 * dF2) / det
        c3 = (A12 * dF1 + A22 * dF2) / det

        c[:, j] = [c0, c1, c2, c3]

    np.savez(RESULTS / 'spectral_coeffs.npz', wl_nm=wl_nm, c0=c[0], c1=c[1], c2=c[2], c3=c[3])
    print("Saved spectral_coeffs.npz")

    # Validate: at 550 nm (index 170)
    idx = 170
    F_approx = c[0, idx]*b0 + c[1, idx]*b1 + c[2, idx]*b2 + c[3, idx]*b3
    F_exact = np.array([ft.fresnel(cos_theta, eta[idx], kappa[idx]) for cos_theta in cT])
    max_err = np.max(np.abs(F_approx - F_exact))
    
    print(f"Validation at {wl_nm[idx]} nm:")
    print(f"  Max absolute error: {max_err:.6f}")
    assert max_err < 0.01
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.degrees(np.arccos(cT)), F_exact, label='Exact Fresnel (Gold @ 550nm)', color='k', linewidth=2)
    plt.plot(np.degrees(np.arccos(cT)), F_approx, label='Belcour Approx', linestyle='--')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Reflectance F')
    plt.title(f'Gold Fresnel Approximation at {wl_nm[idx]} nm')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS / 'coeffs_plot.png')
    print("Saved validation plot to results/coeffs_plot.png")

if __name__ == '__main__':
    main()
