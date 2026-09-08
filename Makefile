# Reduced conductor Fresnel -- validation pipeline.
#   make venv          create .venv and install the pinned dependencies (Mitsuba 3 included)
#   make precompute    IOR -> basis -> coefficients -> CMF -> Phi           (seconds)
#   make check         convention check + Godot export with its checks     (exit 1 on failure)
#   make audit         offline tint dissection (output also saved to results/audit_tint.txt)
#   make splitsum      split-sum LUT, 256x256x4096 Monte Carlo             (minutes, all cores)
#   make illuminants   scene/uffizi_gray.exr and the D65 / FL1 SPD files
#   make render        the 12 Mitsuba renders (4 illuminants x 3 paths)   (long)
#   make compare       the 4 comparison figures + Delta-E numbers
#   make all           everything above, in dependency order
#   make doc           build doc/main.pdf (needs latexmk; or upload doc/ to Overleaf)

PY ?= .venv/bin/python
S   = scripts

.PHONY: all venv precompute check audit splitsum illuminants render compare figures doc clean clean-results

all: precompute check audit splitsum illuminants render compare

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

precompute:
	$(PY) $(S)/load_ior.py
	$(PY) $(S)/extract_basis.py
	$(PY) $(S)/compute_coeffs.py
	$(PY) $(S)/compute_cmf.py
	$(PY) $(S)/compute_phi.py

check:
	$(PY) $(S)/check_conventions.py
	$(PY) $(S)/export_godot.py

audit:
	$(PY) $(S)/audit_tint.py | tee results/audit_tint.txt

splitsum:
	$(PY) $(S)/compute_split_sum.py

illuminants:
	$(PY) $(S)/make_illuminants.py

# Four illuminant settings x three renderers.  Uffizi (RGB HDRI), its achromatic
# version, and two natively spectral constant emitters (D65 smooth, FL1 spiky).
render:
	$(PY) $(S)/render_reference.py
	$(PY) $(S)/render_ours.py
	$(PY) $(S)/render_ours_direct.py
	$(PY) $(S)/render_reference.py   --out reference_gray   --param envfile=uffizi_gray.exr
	$(PY) $(S)/render_ours.py        --out ours_gray        --param envfile=uffizi_gray.exr
	$(PY) $(S)/render_ours_direct.py --out ours_direct_gray --param envfile=uffizi_gray.exr
	$(PY) $(S)/render_reference.py   --scene sphere_spd.xml             --out reference_d65
	$(PY) $(S)/render_ours.py        --scene sphere_ours_spd.xml        --out ours_d65
	$(PY) $(S)/render_ours_direct.py --scene sphere_ours_direct_spd.xml --out ours_direct_d65
	$(PY) $(S)/render_reference.py   --scene sphere_spd.xml             --out reference_fl1   --param illum=illum_fl1.spd
	$(PY) $(S)/render_ours.py        --scene sphere_ours_spd.xml        --out ours_fl1        --param illum=illum_fl1.spd
	$(PY) $(S)/render_ours_direct.py --scene sphere_ours_direct_spd.xml --out ours_direct_fl1 --param illum=illum_fl1.spd

compare:
	$(PY) $(S)/compare.py
	$(PY) $(S)/compare.py --suffix _gray
	$(PY) $(S)/compare.py --suffix _d65
	$(PY) $(S)/compare.py --suffix _fl1

figures:
	$(MAKE) -C doc figures

doc: figures
	$(MAKE) -C doc

clean:
	rm -rf $(S)/__pycache__
	$(MAKE) -C doc clean

clean-results:
	rm -f results/*.exr results/*.png results/*.npz results/*.npy results/*.json results/*.txt scene/uffizi_gray.exr
