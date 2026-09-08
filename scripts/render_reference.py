"""
Ground truth: a full spectral render (Mitsuba 3, llvm_ad_spectral) of a
gold GGX sphere (alpha = 0.05) lit by an environment, with Mitsuba's own
spectral conductor Fresnel.  The scene XML fixes the sampler (2048 spp) and
film (1024 x 1024).

    python scripts/render_reference.py                              # Uffizi HDRI
    python scripts/render_reference.py --out reference_gray --param envfile=uffizi_gray.exr
    python scripts/render_reference.py --scene sphere_spd.xml --out reference_d65
    python scripts/render_reference.py --scene sphere_spd.xml --out reference_fl1 --param illum=illum_fl1.spd

Output: results/<out>.exr and a tonemapped results/<out>.png
"""
import argparse

import mitsuba as mi
import imageio.v3 as iio
import numpy as np

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


parser = argparse.ArgumentParser()
parser.add_argument('--scene', default=str(SCENE / 'sphere.xml'))
parser.add_argument('--out', default='reference', help='basename under results/')
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
