"""
Split-sum LUT — vectorized + parallelized (joblib).

Estimates:
    b_bar_i(cos_theta_o, alpha) = E[b_i(wi.h) * G1(wi, alpha) * (wi.h / wo.h)]
using GGX visible-NDF importance sampling (Heitz 2018).

All 4096 Halton samples processed simultaneously via NumPy broadcasting.
256 NdotV rows dispatched to all CPU cores via joblib.

Output: results/split_sum.npy  shape (256, 256, 4) float32,
        results/split_sum_plot.png
"""

import numpy as np
from scipy.stats import qmc
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time

from _paths import RESULTS


# ---------------------------------------------------------------------------
# GGX helpers
# ---------------------------------------------------------------------------

def G1_smith_ggx(cos_theta, alpha):
    a2  = alpha ** 2
    ct2 = cos_theta ** 2
    return 2.0 * cos_theta / (cos_theta + np.sqrt(a2 + (1.0 - a2) * ct2))


def ggx_vndf_sample_batch(wo, alpha, halton):
    """
    Sample GGX visible NDF for fixed (wo, alpha) and N Halton samples.
    Returns h : (N, 3).
    """
    # 1. Stretch
    wh = np.array([alpha * wo[0], alpha * wo[1], wo[2]])
    wh = wh / np.linalg.norm(wh)

    # 2. Orthonormal basis
    lensq = wh[0]**2 + wh[1]**2
    if lensq > 1e-10:
        T1 = np.array([-wh[1], wh[0], 0.0]) / np.sqrt(lensq)
    else:
        T1 = np.array([1.0, 0.0, 0.0])
    T2 = np.cross(wh, T1)

    # 3. Sample disk
    u1, u2 = halton[:, 0], halton[:, 1]
    r   = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    t1  = r * np.cos(phi)
    t2  = r * np.sin(phi)
    s   = 0.5 * (1.0 + wh[2])
    t2  = (1.0 - s) * np.sqrt(np.clip(1.0 - t1**2, 0.0, None)) + s * t2

    # 4. Project to hemisphere
    nh_z = np.sqrt(np.clip(1.0 - t1**2 - t2**2, 0.0, None))
    nh   = t1[:, None] * T1 + t2[:, None] * T2 + nh_z[:, None] * wh

    # 5. Unstretch
    h = nh * np.array([alpha, alpha, 1.0])
    h[:, 2] = np.clip(h[:, 2], 0.0, None)
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h = h / np.where(norms > 0, norms, 1.0)
    return h


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

def process_ndotv_row(i_ndotv, N_rough, N_samples, halton,
                      cT, b_arrs):
    cos_theta_o = max((i_ndotv + 0.5) / 256, 1e-4)
    wo = np.array([np.sqrt(max(0.0, 1.0 - cos_theta_o**2)), 0.0, cos_theta_o])

    # Build interpolators locally (needed for joblib pickling)
    b_interps = [
        lambda x: np.ones_like(x),
        interp1d(cT, b_arrs[1], kind='linear',
                 fill_value=(b_arrs[1][0], b_arrs[1][-1]), bounds_error=False),
        interp1d(cT, b_arrs[2], kind='linear',
                 fill_value=(b_arrs[2][0], b_arrs[2][-1]), bounds_error=False),
        interp1d(cT, b_arrs[3], kind='linear',
                 fill_value=(b_arrs[3][0], b_arrs[3][-1]), bounds_error=False),
    ]

    row = np.zeros((N_rough, 4), dtype=np.float32)

    for i_rough in range(N_rough):
        alpha = max((i_rough + 0.5) / 256, 0.001)

        h        = ggx_vndf_sample_batch(wo, alpha, halton)   # (N, 3)
        wo_dot_h = h @ wo                                       # (N,)
        wi       = 2.0 * wo_dot_h[:, None] * h - wo            # (N, 3)
        wi_z     = wi[:, 2]
        wi_dot_h = (wi * h).sum(axis=1)                        # (N,)

        valid  = (wi_z > 0) & (wi_dot_h > 0) & (wo_dot_h > 0)
        weight = np.zeros(N_samples)
        weight[valid] = (
            G1_smith_ggx(wi_z[valid], alpha)
            * wi_dot_h[valid] / wo_dot_h[valid]
        )

        for k in range(4):
            b_vals         = b_interps[k](wi_dot_h)
            row[i_rough, k] = (b_vals * weight).sum() / N_samples

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    basis_data = np.load(RESULTS / 'basis_functions.npz')
    cT     = basis_data['cT']
    b_arrs = [basis_data['b0'], basis_data['b1'],
              basis_data['b2'], basis_data['b3']]

    N_samples = 4096
    N_ndotv   = 256
    N_rough   = 256

    halton = qmc.Halton(d=2, scramble=True, seed=0).random(N_samples)

    print(f"Computing split-sum LUT ({N_ndotv}x{N_rough}, {N_samples} spp)...")
    t0 = time.time()

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_ndotv_row)(i, N_rough, N_samples, halton, cT, b_arrs)
        for i in range(N_ndotv)
    )

    split_sum = np.stack(results, axis=0).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    np.save(RESULTS / 'split_sum.npy', split_sum)
    print(f"Saved results/split_sum.npy  shape: {split_sum.shape}")

    # --- Validation plots ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = ['b0 (albedo, F=1)', 'b1', 'b2', 'b3']
    for k in range(4):
        im = axes[k].imshow(split_sum[:, :, k], origin='lower',
                            extent=[0, 1, 0, 1], aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=axes[k])
        axes[k].set_xlabel('Roughness alpha')
        axes[k].set_ylabel('cos(theta_o)')
        axes[k].set_title(titles[k])
    plt.suptitle('Split-sum LUT: b_bar_i(NdotV, alpha)', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / 'split_sum_plot.png', dpi=120, bbox_inches='tight')
    print("Validation plot -> results/split_sum_plot.png")

    # Numeric sanity checks
    v_near1 = split_sum[255, 0, 0]
    v_low   = split_sum[0, 255, 0]
    print(f"\nSanity: split_sum[255, 0,  0] NdotV~1 alpha~0  b0 = {v_near1:.4f}  (expect ~1.0)")
    print(f"Sanity: split_sum[0,   255, 0] NdotV~0 alpha~1  b0 = {v_low:.4f}  (expect <1.0)")
    assert v_near1 > 0.95
    assert v_low < 1.0


if __name__ == '__main__':
    main()
