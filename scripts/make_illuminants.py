"""
Generate the deconfounded illuminant assets used by the _gray, _d65 and _fl1
experiments:

  1. scene/uffizi_gray.exr — luminance-only (achromatic) version of uffizi.hdr.
     Keeps the full angular structure of the envmap but removes its chroma, so
     the Jakob-Hanika uplift (reference) and the dual-basis metamer (ours)
     agree to first order: an achromatic pixel maps to a near-flat spectrum
     under both.
  2. scene/illum_d65.spd, scene/illum_fl1.spd — natively spectral illuminants
     (CIE D65 smooth, CIE FL1 spiky fluorescent) for `constant` emitters.
     Zero uplift ambiguity: reference and reduced paths see the same SPD.
     Normalized so that photometric luminance Y = 1.

Requires results/spectral_coeffs.npz (for the wavelength grid).
"""
import numpy as np
import colour

from _paths import SCENE, RESULTS

# ── 1. Grayscale envmap ───────────────────────────────────────────────────────
# Load with MITSUBA's own HDR loader: imageio decodes Radiance RGBE on a 0-255
# scale while Mitsuba decodes to ~0-16 — using imageio here made the gray map
# ~44x too bright (blown-out renders).
import mitsuba as mi
mi.set_variant('scalar_rgb')
img = np.array(mi.Bitmap(str(SCENE / 'uffizi.hdr'))).astype(np.float32)
print(f"uffizi.hdr (mitsuba loader): {img.shape}, range [{img.min():.3f}, {img.max():.3f}]")

# Rec.709 luminance, replicated to R=G=B
Y = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
gray = np.repeat(Y[..., None], 3, axis=-1).astype(np.float32)

mi.Bitmap(gray).write(str(SCENE / 'uffizi_gray.exr'))
print("Saved scene/uffizi_gray.exr")

# ── 2. Spectral illuminants ───────────────────────────────────────────────────
# Same wavelength grid as the rest of the pipeline
wl_nm = np.load(RESULTS / 'spectral_coeffs.npz')['wl_nm']

# CIE 1931 ybar for luminance normalization (Mitsuba's own table, matching
# compute_cmf.py / check_conventions.py conventions)
mi.set_variant('scalar_spectral')
ybar = np.array([mi.cie1931_xyz(float(w))[1] for w in wl_nm])
dl = np.gradient(wl_nm)

# Mitsuba's spectral film divides by the CIE Y integral (∫ybar dλ ≈ 106.857),
# so "film luminance = 1" requires ∫SPD·ybar dλ = ∫ybar dλ, not = 1.
Y_integral = (ybar * dl).sum()
print(f"CIE Y integral on our grid: {Y_integral:.4f}")

for name in ('D65', 'FL1'):
    sd = colour.SDS_ILLUMINANTS[name]
    vals = np.interp(wl_nm, sd.wavelengths, sd.values, left=0.0, right=0.0)
    # Normalize so the FILM sees luminance Y = 1
    vals = vals / (vals * ybar * dl).sum() * Y_integral
    path = SCENE / f'illum_{name.lower()}.spd'
    with open(path, 'w') as f:
        for w, v in zip(wl_nm, vals):
            f.write(f"{w:.1f} {v:.8e}\n")
    print(f"Saved {path}  (film Y={(vals * ybar * dl).sum() / Y_integral:.6f}, peak={vals.max():.4f})")
