"""
The real-time path: an RGB render (llvm_ad_rgb) in which a custom `direct`
integrator keeps the reduced Fresnel as a genuine 3x3 matrix

        M_F(cosT) = sum_i b_i(cosT) * Phi_i          (Phi_i are 3x3, XYZ-space)

and applies it to the *actual* radiance Mitsuba transports at the single bounce
of a `direct` integrator:  L_out = M_F @ L_in.

Design:
  * The scene sphere is a stock `roughconductor` with dummy IOR (F~1). It is used
    ONLY as the achromatic GGX geometry term + VNDF importance sampler.
  * This custom integrator owns Phi_i, evaluates the geometry via the BSDF, and
    replaces the (component-wise) Fresnel multiply with the full matrix product.
  * Radiance in llvm_ad_rgb is linear sRGB, while Phi_i act on XYZ, so we
    pre-fold the colour-space change:  Psi_i = M_rgb<-xyz @ Phi_i @ M_xyz<-rgb.

This is the illuminant-free counterpart of the reduction, and the
single-bounce case of the general rule (throughput becomes a 3x3 matrix).

    python scripts/render_ours_direct.py [--scene sphere_ours_direct_spd.xml] [--out ours_direct_d65] [--param k=v]

Output: results/<out>.exr and a tonemapped results/<out>.png
"""
import argparse
import time

import mitsuba as mi
import drjit as dr
import numpy as np
import imageio.v3 as iio

from _paths import SCENE, RESULTS

mi.set_variant('llvm_ad_rgb')

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

# -- Load precomputations -----------------------------------------------------
print("Loading precomputations...")
basis = np.load(RESULTS / 'basis_functions.npz')
b1_arr = mi.Float(basis['b1'].astype(np.float32))
b2_arr = mi.Float(basis['b2'].astype(np.float32))
b3_arr = mi.Float(basis['b3'].astype(np.float32))

phi_data = np.load(RESULTS / 'Phi.npz')

# Colour-space matrices. Phi_i act in XYZ; Mitsuba rgb radiance is linear sRGB.
XYZ_TO_sRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
sRGB_TO_XYZ = np.linalg.inv(XYZ_TO_sRGB)

# Pre-fold colour conversion into each basis matrix so we can work directly on
# the linear-sRGB radiance that the integrator carries.
PSI = [ (XYZ_TO_sRGB @ phi_data[f'Phi{i}'] @ sRGB_TO_XYZ).astype(np.float64)
        for i in range(4) ]


def gather_curve(arr, cosT):
    idx = dr.clip(mi.UInt32(cosT * 999.0 + 0.5), 0, 999)
    return dr.gather(mi.Float, arr, idx)


def reduced_fresnel_apply(cosT, L):
    """Return  M_F(cosT) @ L  in linear sRGB, where L is a mi.Color3f radiance."""
    B = [mi.Float(1.0),                 # b0 == 1
         gather_curve(b1_arr, cosT),
         gather_curve(b2_arr, cosT),
         gather_curve(b3_arr, cosT)]
    Lc = [L.x, L.y, L.z]
    out = []
    for a in range(3):
        acc = mi.Float(0.0)
        for b in range(3):
            m_ab = (B[0] * float(PSI[0][a][b]) + B[1] * float(PSI[1][a][b])
                    + B[2] * float(PSI[2][a][b]) + B[3] * float(PSI[3][a][b]))
            acc = acc + m_ab * Lc[b]
        out.append(dr.maximum(acc, 0.0))   # clamp tiny out-of-gamut negatives
    return mi.Color3f(out[0], out[1], out[2])


def mis_weight(pdf_a, pdf_b):
    """Power (squared) heuristic, matching Mitsuba's `direct` integrator."""
    a2 = dr.sqr(pdf_a)
    return dr.select(pdf_a > 0.0, a2 / (a2 + dr.sqr(pdf_b)), 0.0)


# ── Custom direct integrator ──────────────────────────────────────────────────
class ReducedDirect(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        ctx = mi.BSDFContext()
        active = mi.Bool(active)

        si = scene.ray_intersect(ray, active)
        result = mi.Color3f(0.0)

        # (1) Directly visible emission (background envmap; 0 on the sphere).
        result += si.emitter(scene, active).eval(si, active)

        active_s = active & si.is_valid()
        bsdf = si.bsdf(ray)

        # (2) Emitter sampling (NEE) --------------------------------------------
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(active_s), True, active_s)
        active_em = active_s & (ds.pdf > 0.0)

        wo_local = si.to_local(ds.d)
        f_geom   = bsdf.eval(ctx, si, wo_local, active_em)      # achromatic geom
        g        = f_geom.x
        bsdf_pdf = bsdf.pdf(ctx, si, wo_local, active_em)
        mis_e    = mis_weight(ds.pdf, bsdf_pdf)

        H     = dr.normalize(si.wi + wo_local)
        cosT  = dr.clip(dr.dot(si.wi, H), 0.0, 1.0)
        L_em  = reduced_fresnel_apply(cosT, em_weight)
        result += dr.select(active_em, g * mis_e * L_em, mi.Color3f(0.0))

        # (3) BSDF sampling -----------------------------------------------------
        bs, bsdf_weight = bsdf.sample(
            ctx, si, sampler.next_1d(active_s), sampler.next_2d(active_s), active_s)
        active_b = active_s & (bs.pdf > 0.0)
        g2 = bsdf_weight.x

        ray_b = si.spawn_ray(si.to_world(bs.wo))
        si_b  = scene.ray_intersect(ray_b, active_b)
        L_in  = si_b.emitter(scene, active_b).eval(si_b, active_b)  # envmap hit

        ds_b   = mi.DirectionSample3f(scene, si_b, si)
        em_pdf = scene.pdf_emitter_direction(si, ds_b, active_b)
        mis_b  = mis_weight(bs.pdf, em_pdf)

        Hb    = dr.normalize(si.wi + bs.wo)
        cosTb = dr.clip(dr.dot(si.wi, Hb), 0.0, 1.0)
        L_bs  = reduced_fresnel_apply(cosTb, L_in)
        result += dr.select(active_b, g2 * mis_b * L_bs, mi.Color3f(0.0))

        return (result, si.is_valid(), [])

mi.register_integrator("reduced_direct", lambda props: ReducedDirect(props))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default=str(SCENE / 'sphere_ours_direct.xml'))
    parser.add_argument('--out', default='ours_direct', help='basename under results/')
    parser.add_argument('--param', action='append', default=[],
                        help='scene XML parameter, k=v (e.g. envfile=uffizi_gray.exr)')
    args = parser.parse_args()
    params = dict(kv.split('=', 1) for kv in args.param)

    print(f"Loading scene {args.scene} {params}...")
    scene = mi.load_file(_resolve_scene(args.scene), **params)

    spp = scene.sensors()[0].sampler().sample_count()
    print(f"Rendering ({spp} spp)...")
    t0 = time.time()
    img = mi.render(scene)
    print(f"Done rendering in {time.time() - t0:.2f} s.")
    _save_outputs(img, args.out)

if __name__ == '__main__':
    main()
