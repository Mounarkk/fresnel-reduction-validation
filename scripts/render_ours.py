"""
Spectral render with the ANGULAR decomposition only.  A custom Mitsuba BSDF
replaces the exact conductor Fresnel of a GGX sphere by its 4-basis
reconstruction  F ~= c_0 + c_1 b_1 + c_2 b_2 + c_3 b_3, evaluated per
wavelength of each ray packet.  No spectral reduction is involved: this
isolates the error of the Belcour et al. (2020) basis fit alone.

The geometry term and importance sampling come from an inner stock
`roughconductor` with a dummy IOR (eta = 0, k = 1e5, so F ~ 1).

    python scripts/render_ours.py [--scene sphere_ours_spd.xml] [--out ours_d65] [--param k=v]

Output: results/<out>.exr and a tonemapped results/<out>.png
"""
import argparse

import mitsuba as mi
import drjit as dr
import numpy as np
import imageio.v3 as iio

from _paths import SCENE, RESULTS

mi.set_variant('llvm_ad_spectral')

def _resolve_scene(path_str):
    """Accept an absolute path, a path relative to the cwd, or a bare scene name."""
    from pathlib import Path
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())
    q = SCENE / p.name
    if q.exists():
        return str(q)
    raise FileNotFoundError(f"scene not found: {path_str} (also tried {q})")


def _save_outputs(img, out_name):
    """Write results/<out>.exr and a Reinhard-tonemapped results/<out>.png."""
    RESULTS.mkdir(exist_ok=True)
    mi.util.write_bitmap(str(RESULTS / f'{out_name}.exr'), img)
    print(f"Saved results/{out_name}.exr")
    arr = np.nan_to_num(np.array(img), nan=0.0)
    arr_tm = np.clip(arr / (1.0 + arr), 0.0, 1.0)
    arr_u8 = (arr_tm ** (1 / 2.2) * 255).astype(np.uint8)
    iio.imwrite(RESULTS / f'{out_name}.png', arr_u8)
    print(f"Saved results/{out_name}.png")
    print(f"Image stats: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

# -- Basis functions ----------------------------------------------------------
d = np.load(RESULTS / 'basis_functions.npz')
b1_np = d['b1'].astype(np.float32)
b2_np = d['b2'].astype(np.float32)
b3_np = d['b3'].astype(np.float32)
b1_arr = mi.Float(b1_np)
b2_arr = mi.Float(b2_np)
b3_arr = mi.Float(b3_np)

# Probe angles come from the basis extraction (single source of truth).
ct1 = float(d['probe_ct1'])
ct2 = float(d['probe_ct2'])
i1  = int(d['probe_i1'])
i2  = int(d['probe_i2'])

A11 = float(b3_np[i2])
A12 = -float(b2_np[i2])
A21 = -float(b3_np[i1])
A22 = float(b2_np[i1])
det = A11 * A22 - A12 * A21

def gather_curve(arr, cosT):
    # +0.5 = round to nearest table entry (matches render_ours_direct.py)
    idx = dr.clip(mi.UInt32(cosT * 999.0 + 0.5), 0, 999)
    return dr.gather(mi.Float, arr, idx)

# ── Custom BSDF ───────────────────────────────────────────────────────────────
class OursBSDF(mi.BSDF):
    def __init__(self, props):
        super().__init__(props)
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

        # props.get_texture() returns an instantiated mitsuba.Texture with .eval_1()
        self.eta_tex = props.get_texture('eta', 1.0)
        self.k_tex   = props.get_texture('k',   1.0)

        # Inner roughconductor with dummy IOR that gives F=1 everywhere.
        # This isolates the geometric term D*G/(4*cos_i*cos_o) so we can multiply F_approx.
        self.rc = mi.load_dict({
            'type': 'roughconductor',
            'distribution': props.get('distribution', 'ggx'),
            'alpha': props.get('alpha', 0.01),
            'eta': {'type': 'spectrum', 'value': 0.0},
            'k':   {'type': 'spectrum', 'value': 100000.0},
        })

    def _eval_fresnel_approx(self, si, cosT):
        """Evaluate the 4-basis Fresnel approximation.
        Uses eval() which returns a mi.Spectrum (4 values per ray packet).
        We calculate the Fresnel approximation per-channel.
        """
        eta_val = self.eta_tex.eval(si, active=True)
        k_val   = self.k_tex.eval(si, active=True)
        
        # Use the SVD-extracted b1 (NOT Schlick's (1-cosT)^5) so the basis is
        # consistent with compute_coeffs.py / compute_phi.py.
        B1 = gather_curve(b1_arr, cosT)
        B2 = gather_curve(b2_arr, cosT)
        B3 = gather_curve(b3_arr, cosT)

        F_result = mi.Spectrum(0.0)
        # Mitsuba 3 spectral variants carry 4 wavelengths per ray packet
        for i in range(4):
            c = mi.Complex2f(eta_val[i], k_val[i])

            # Belcour 4-basis decomposition
            C0 = mi.fresnel_conductor(mi.Float(1.0), c)
            C1 = dr.maximum(1.0 - C0, 0.0)

            F_ct1 = mi.fresnel_conductor(mi.Float(ct1), c)
            F_ct2 = mi.fresnel_conductor(mi.Float(ct2), c)

            # Probe residuals must use the same b1 as the basis being fit
            # (matches compute_coeffs.py: dF = F(ct) - (c0 + c1*b1[i])).
            dF1 = F_ct1 - (C0 + C1 * float(b1_np[i1]))
            dF2 = F_ct2 - (C0 + C1 * float(b1_np[i2]))

            C2 = (A11 * dF1 + A21 * dF2) / det
            C3 = (A12 * dF1 + A22 * dF2) / det

            F_approx = C0 + C1 * B1 + C2 * B2 + C3 * B3
            F_result[i] = dr.maximum(F_approx, 0.0)

        return F_result

    def eval(self, ctx, si, wo, active):
        val  = self.rc.eval(ctx, si, wo, active)
        H    = dr.normalize(si.wi + wo)
        cosT = dr.clip(dr.dot(si.wi, H), 0.0, 1.0)
        return val * self._eval_fresnel_approx(si, cosT)

    def sample(self, ctx, si, sample1, sample2, active):
        bs, weight = self.rc.sample(ctx, si, sample1, sample2, active)
        H    = dr.normalize(si.wi + bs.wo)
        cosT = dr.clip(dr.dot(si.wi, H), 0.0, 1.0)
        return bs, weight * self._eval_fresnel_approx(si, cosT)

    def pdf(self, ctx, si, wo, active):
        return self.rc.pdf(ctx, si, wo, active)

    def to_string(self):
        return "OursBSDF[]"

mi.register_bsdf("ours", lambda props: OursBSDF(props))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=str(SCENE / 'sphere_ours.xml'))
    parser.add_argument('--out', default='ours', help='basename under results/')
    parser.add_argument('--param', action='append', default=[],
                        help='scene XML parameter, k=v (e.g. envfile=uffizi_gray.exr)')
    args = parser.parse_args()
    params = dict(kv.split('=', 1) for kv in args.param)

    print(f"Loading scene {args.scene} {params}...")
    scene = mi.load_file(_resolve_scene(args.scene), **params)

    spp = scene.sensors()[0].sampler().sample_count()
    print(f"Rendering ({spp} spp)...")
    img = mi.render(scene)
    print("Done rendering.")
    _save_outputs(img, args.out)

if __name__ == '__main__':
    main()