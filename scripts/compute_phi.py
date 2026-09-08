"""
Reduce each spectral coefficient function with the dual basis:

    Phi_i = S^T diag(c_i(lambda)) Sb        (3x3, i = 0..3)

These four matrices are the only per-material data the runtime needs; the
angular dependence lives entirely in the scalar weights b_i(cos theta).

Output: results/Phi.npz  (Phi0..Phi3, float32)
"""
import numpy as np
from scipy.interpolate import interp1d

from _paths import RESULTS

def main():
    # Load c from spectral_coeffs.npz
    coeffs_data = np.load(RESULTS / 'spectral_coeffs.npz')
    c0 = coeffs_data['c0']
    c1 = coeffs_data['c1']
    c2 = coeffs_data['c2']
    c3 = coeffs_data['c3']
    c = np.array([c0, c1, c2, c3]) # shape (4, 401)
    
    # Load S and Sb from cmf.npz
    cmf_data = np.load(RESULTS / 'cmf.npz')
    S = cmf_data['S']   # shape (401, 3)
    Sb = cmf_data['Sb'] # shape (401, 3)
    
    # Compute Phi for i in 0..3
    Phi = np.zeros((4, 3, 3), dtype=np.float32)
    for i in range(4):
        # Phi[i] = S.T @ diag(c[i]) @ Sb = (S.T * c[i]) @ Sb
        Phi[i] = (S.T * c[i]) @ Sb
        
    out_path = RESULTS / 'Phi.npz'
    np.savez(out_path, Phi0=Phi[0], Phi1=Phi[1], Phi2=Phi[2], Phi3=Phi[3])
    print(f"Saved Phi0..Phi3 to {out_path}")
    
    # Validation: pick cos_theta = 0.5
    basis_data = np.load(RESULTS / 'basis_functions.npz')
    cT = basis_data['cT']
    b0 = basis_data['b0']
    b1 = basis_data['b1']
    b2 = basis_data['b2']
    b3 = basis_data['b3']
    b_all = [b0, b1, b2, b3]
    
    bbar = np.zeros(4)
    for i in range(4):
        interpolator = interp1d(cT, b_all[i], kind='linear', fill_value='extrapolate')
        bbar[i] = interpolator(0.5)
        
    M_F = np.sum([bbar[i] * Phi[i] for i in range(4)], axis=0)
    
    # Apply to a flat white illuminant
    c_I = S.T @ np.ones(len(c0))
    xyz_color = M_F @ c_I
    
    print(f"Validation: XYZ color for flat white at cos_theta=0.5: {xyz_color}")
    print("Gold is expected to be yellow-orange: Y > Z, with 0 < Y < Y_in.")
    Y_in = c_I[1]
    assert 0.0 < xyz_color[1] < Y_in
    assert xyz_color[2] < xyz_color[1]

if __name__ == '__main__':
    main()
