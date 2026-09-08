"""
Exact Fresnel reflectance for dielectrics and conductors.

Reduced from the Fresnel helper module distributed with the supplemental code
of Belcour, Bati & Barla (2020); only the functions this pipeline calls are
kept, with their numerics unchanged.  The dielectric branch follows the
formulation on S. Lagarde's blog, as in the original module.

    fresnel(cos_theta, eta, kappa=0.0, polarized=False)
        Unpolarised Fresnel reflectance (or the (Rs, Rp) pair) at a given
        cosine of the incident angle, for complex IOR  eta + i*kappa.
"""
import numpy as np


def polarized_fresnel(CosTheta, Eta):
    """Dielectric Fresnel (Rs, Rp).  Eta = eta_t / eta_i.  Total internal
    reflection is returned as 1."""
    SinTheta2 = 1 - CosTheta * CosTheta
    temp = 1 - (SinTheta2 / (Eta * Eta))
    tempo = np.where(temp < 0, 1.0, temp)
    t0 = np.sqrt(tempo)
    t1 = Eta * t0
    t2 = Eta * CosTheta
    rs = (CosTheta - t1) / (CosTheta + t1)
    rp = (t0 - t2) / (t0 + t2)
    return np.where(temp < 0, 1.0, rs * rs), np.where(temp < 0, 1.0, rp * rp)


def fresnel_dielectric_conductor(Eta, Etak, CosTheta, polarized=False):
    """Conductor Fresnel for complex IOR  Eta + i*Etak."""
    CosTheta2 = CosTheta * CosTheta
    SinTheta2 = 1 - CosTheta2
    Eta2 = Eta * Eta
    Etak2 = Etak * Etak

    t0 = Eta2 - Etak2 - SinTheta2
    a2plusb2 = np.sqrt(t0 * t0 + 4 * Eta2 * Etak2)
    t1 = a2plusb2 + CosTheta2
    temp = 0.5 * (a2plusb2 + t0)
    temp = np.where(temp > 0.0, temp, 1.0)

    a = np.sqrt(temp)
    t2 = 2 * a * CosTheta
    Rs = (t1 - t2) / (t1 + t2)

    t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2
    t4 = t2 * SinTheta2
    Rp = Rs * (t3 - t4) / (t3 + t4)

    if polarized:
        return Rs, Rp
    return 0.5 * (Rs + Rp)


def fresnel(CosTheta, Eta, kappa=0.0, polarized=False):
    """Unpolarised Fresnel reflectance (or (Rs, Rp) if polarized=True)."""
    CosTheta = np.maximum(CosTheta, 1.0e-8)
    if kappa == 0.0:
        Rs, Rp = polarized_fresnel(CosTheta, Eta)
        if polarized:
            return Rs, Rp
        return 0.5 * (Rs + Rp)
    return fresnel_dielectric_conductor(Eta, kappa, CosTheta, polarized)
