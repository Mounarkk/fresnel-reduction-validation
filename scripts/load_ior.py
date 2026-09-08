"""
Resample the gold complex IOR  eta(lambda) + i kappa(lambda)  from the
two-block CSV in data/ (wavelength in micrometres; first block n, second
block k) onto the 380-780 nm, 1 nm grid used by every later stage.

Output: results/gold_ior.npz  (wl_nm, eta, kappa)
"""
import io

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from _paths import DATA, RESULTS

def main():
    print("Loading and interpolating gold IOR data...")
    # The CSV has two blocks separated by a blank line
    # Block 1: wl(µm), n
    # Block 2: wl(µm), k
    RESULTS.mkdir(exist_ok=True)
    with open(DATA / 'gold_refractive_index.csv', 'r') as f:
        raw = f.read()
    
    blocks = raw.strip().split('\n\n')
    
    df_n = pd.read_csv(io.StringIO(blocks[0]))
    df_k = pd.read_csv(io.StringIO(blocks[1]))
    
    # Target wavelength grid: 380–780 nm at 1 nm steps (convert µm → nm)
    wl_nm = np.arange(380, 781, 1, dtype=float)
    
    # The original wavelengths are in µm, so we multiply by 1000 to get nm
    eta_interp = interp1d(df_n['wl'].values * 1000.0, df_n['n'].values,
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    kap_interp = interp1d(df_k['wl'].values * 1000.0, df_k['k'].values,
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    
    eta = eta_interp(wl_nm)
    kappa = kap_interp(wl_nm)
    
    np.savez(RESULTS / 'gold_ior.npz', wl_nm=wl_nm, eta=eta, kappa=kappa)
    print("Saved gold_ior.npz")
    
    # Validation: print gold at ~550 nm (index 170)
    idx_550 = 170
    print(f"Validation at {wl_nm[idx_550]} nm:")
    print(f"  eta   = {eta[idx_550]:.4f} (expected ~0.4)")
    print(f"  kappa = {kappa[idx_550]:.4f} (expected ~2.3)")
    assert 0.25 < eta[idx_550] < 0.50 and 2.0 < kappa[idx_550] < 2.6

if __name__ == '__main__':
    main()
