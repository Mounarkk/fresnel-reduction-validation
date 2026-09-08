"""
Error of the two reduced renders against the spectral reference:
  - per-pixel RMSE on linear HDR values,
  - per-pixel CIEDE2000 on Reinhard-tonemapped sRGB converted to CIELAB,
  - a 2x3 figure: reference | ours (spectral, 4-basis) | ours (reduced matrix)
                            | dE00 spectral            | dE00 reduced matrix

    python scripts/compare.py [--suffix _gray|_d65|_fl1]

Reads  results/reference<sfx>.exr, ours<sfx>.exr, ours_direct<sfx>.exr
Writes results/comparison<sfx>.png
"""
import argparse

import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
import skimage.color
import colour

from _paths import RESULTS

mi.set_variant('scalar_spectral')
_p = argparse.ArgumentParser()
_p.add_argument('--suffix', default='', help="experiment suffix, e.g. _gray, _d65, _fl1 "
                "-> compares reference<sfx>/ours<sfx>/ours_direct<sfx>.exr")
sfx = _p.parse_args().suffix

# ------------------------------------------------------------------
# Load images
# ------------------------------------------------------------------
ref         = np.nan_to_num(np.array(mi.Bitmap(str(RESULTS / f'reference{sfx}.exr'))),   nan=0.0)
ours        = np.nan_to_num(np.array(mi.Bitmap(str(RESULTS / f'ours{sfx}.exr'))),        nan=0.0)
ours_direct = np.nan_to_num(np.array(mi.Bitmap(str(RESULTS / f'ours_direct{sfx}.exr'))), nan=0.0)

print(f"Reference shape: {ref.shape}")
print(f"Ours (Spec) shape: {ours.shape}")
print(f"Ours (Direct) shape: {ours_direct.shape}")

# ------------------------------------------------------------------
# Reinhard tonemapping for perceptual metrics (clip to [0,1])
# ------------------------------------------------------------------
ref_tm      = np.clip(ref      / (1.0 + ref),      0.0, 1.0)
ours_tm     = np.clip(ours     / (1.0 + ours),     0.0, 1.0)
ours_direct_tm = np.clip(ours_direct / (1.0 + ours_direct), 0.0, 1.0)

# ------------------------------------------------------------------
# RMSE (in linear HDR, meaningful for energy error)
# ------------------------------------------------------------------
rmse_spec = np.sqrt(((ref - ours) ** 2).mean(axis=-1))
rmse_rgb  = np.sqrt(((ref - ours_direct) ** 2).mean(axis=-1))

print(f"\nRMSE (Spectral) — mean: {rmse_spec.mean():.4f}  max: {rmse_spec.max():.4f}")
print(f"RMSE (Direct)   — mean: {rmse_rgb.mean():.4f}  max: {rmse_rgb.max():.4f}")

# ------------------------------------------------------------------
# Delta E CIEDE2000 (on tonemapped [0,1] sRGB -> Lab)
# Target: mean < 0.5 (ideal), < 1.0 (acceptable)
# ------------------------------------------------------------------

ref_lab      = skimage.color.rgb2lab(ref_tm)
ours_lab     = skimage.color.rgb2lab(ours_tm)
ours_direct_lab = skimage.color.rgb2lab(ours_direct_tm)

delta_e_spec = colour.difference.delta_E_CIE2000(ref_lab, ours_lab)
delta_e_rgb  = colour.difference.delta_E_CIE2000(ref_lab, ours_direct_lab)

print(f"ΔE00 (Spectral) — mean: {delta_e_spec.mean():.4f}  max: {delta_e_spec.max():.4f}")
print(f"ΔE00 (Direct)   — mean: {delta_e_rgb.mean():.4f}  max: {delta_e_rgb.max():.4f}")

def verdict(de):
    m = de.mean()
    return "PASS ✓ (< 0.5)" if m < 0.5 else ("ACCEPTABLE ~ (< 1.0)" if m < 1.0 else "FAIL ✗ (>= 1.0)")

print(f"Target mean ΔE00 < 0.5 (Spectral): {verdict(delta_e_spec)}")
print(f"Target mean ΔE00 < 0.5 (Direct):   {verdict(delta_e_rgb)}")

# ------------------------------------------------------------------
# Figure Generation
# ------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Top row: Images
axs[0, 0].imshow(np.clip(ref_tm  ** (1/2.2), 0, 1))
axs[0, 0].set_title('Reference (full spectral)', fontsize=12)
axs[0, 0].axis('off')

axs[0, 1].imshow(np.clip(ours_tm ** (1/2.2), 0, 1))
axs[0, 1].set_title('Ours (spectral, 4-basis Fresnel)', fontsize=12)
axs[0, 1].axis('off')

axs[0, 2].imshow(np.clip(ours_direct_tm ** (1/2.2), 0, 1))
axs[0, 2].set_title('Ours (reduced 3x3 matrix, RGB)', fontsize=12)
axs[0, 2].axis('off')

# Bottom row: Errors
axs[1, 0].axis('off') # Leave bottom-left empty

# Cap colorbar at 5.0 so high-variance fireflies don't wash out the heatmap
vmax = min(max(delta_e_spec.max(), delta_e_rgb.max()), 5.0)

im1 = axs[1, 1].imshow(delta_e_spec, cmap='hot', vmin=0, vmax=vmax)
plt.colorbar(im1, ax=axs[1, 1], fraction=0.046, pad=0.04)
axs[1, 1].set_title(f'dE00 spectral (mean={delta_e_spec.mean():.3f})', fontsize=12)
axs[1, 1].axis('off')

im2 = axs[1, 2].imshow(delta_e_rgb, cmap='hot', vmin=0, vmax=vmax)
plt.colorbar(im2, ax=axs[1, 2], fraction=0.046, pad=0.04)
axs[1, 2].set_title(f'dE00 reduced matrix (mean={delta_e_rgb.mean():.3f})', fontsize=12)
axs[1, 2].axis('off')

plt.suptitle(f'Reduced conductor Fresnel, gold sphere: comparison{sfx}', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / f'comparison{sfx}.png', dpi=150, bbox_inches='tight')
print(f"\nSaved results/comparison{sfx}.png")
