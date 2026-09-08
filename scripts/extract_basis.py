"""
Read the four universal angular basis functions  b_0..b_3(cos theta)  of
Belcour, Bati & Barla (2020) from the authors' Mitsuba plugin source
(data/belcour2020/fresnel_curves.hpp, table fresnel_curves[1000][4]) and
record the two probe angles their coefficient solve uses.

Output: results/basis_functions.npz, results/basis_plot.png
"""
import re

import numpy as np
import matplotlib.pyplot as plt

from _paths import DATA, RESULTS

def main():
    print("Extracting basis functions from fresnel_curves.hpp...")
    with open(DATA / 'belcour2020' / 'fresnel_curves.hpp', 'r') as f:
        text = f.read()

    # The table starts at 'float fresnel_curves[1000][4] = {'
    start_idx = text.find('fresnel_curves')
    if start_idx == -1:
        raise ValueError("Could not find 'fresnel_curves' array in the file.")
    
    # Extract all { v0, v1, v2, v3 } entries
    # The regex might capture an extra '{' if not careful, so clean up first
    rows = re.findall(r'\{([^}]+)\}', text[start_idx:])
    
    # Keep only the 1000 data rows
    data = []
    for r in rows:
        # remove any stray opening braces
        r = r.replace('{', '').strip()
        parts = r.split(',')
        if len(parts) >= 4:
            try:
                vals = [float(x.strip()) for x in parts[:4]]
                data.append(vals)
            except ValueError:
                pass
        if len(data) == 1000:
            break

    data = np.array(data)
    assert data.shape == (1000, 4), f"Expected shape (1000, 4), got {data.shape}"

    # The table is indexed by cos_theta in [0,1], row k -> cos_theta = k/999
    cT = np.linspace(0.0, 1.0, 1000, endpoint=True)
    basis0 = data[:, 0]   # always 1.0
    basis1 = data[:, 1]
    basis2 = data[:, 2]
    basis3 = data[:, 3]

    # Probe indices and cos_theta values, as hardcoded in fresnelToCoeffs() of the same file
    i1, ct1 = 120, 0.12012012012012012
    i2, ct2 = 434, 0.4344344344344344

    np.savez(RESULTS / 'basis_functions.npz',
             b0=basis0, b1=basis1, b2=basis2, b3=basis3,
             cT=cT, probe_ct1=ct1, probe_ct2=ct2, probe_i1=i1, probe_i2=i2)
    
    print("Saved basis_functions.npz")

    # Validate: plot all 4 basis curves vs arccos(cT) in degrees
    angles = np.degrees(np.arccos(cT))
    plt.figure(figsize=(10, 6))
    plt.plot(angles, basis0, label='b0 (Mean F)', linestyle='--')
    plt.plot(angles, basis1, label='b1 (F0 peak)')
    plt.plot(angles, basis2, label='b2 (Oscillatory 1)')
    plt.plot(angles, basis3, label='b3 (Oscillatory 2)')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Basis Value')
    plt.title('Extracted Fresnel Basis Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS / 'basis_plot.png')
    print("Saved validation plot to results/basis_plot.png")

if __name__ == '__main__':
    main()
