# Determine the amount of energy absorbed by a slab near a point source. This
# assumes the following:
#
# - No self-heating (no energy absorbed from re-emitted radiation)
# - No scattering (very optically thin)

import numpy as np

from astropy.utils.console import ProgressBar

xs = 0.
ys = 0.
zs = 4.

xmin = -5.
xmax = +5.
ymin = -5.
ymax = +5.
zmin = -5.
zmax = -2.

N = 100000
REALIZATIONS = 1000
TAU = 1e-10

# Read in spectrum and construct CDF
from scipy.integrate import cumtrapz
lam, fnu, flam = np.loadtxt('data/BB_T10000_L100000.dat', unpack=True)
cdf = cumtrapz(flam, x=lam, initial=0)
cdf /= cdf[-1]

# Read in opacities and determine optical depth per unit length as a function of wavelength
dust_wav, cabs = np.loadtxt('data/ZDA_BARE_GR_S_Effective.dat', unpack=True, usecols=[0, 1], skiprows=4)
cabs_1mu = np.interp(1., dust_wav, cabs)
tau_unit = cabs / cabs_1mu / (zmax - zmin) * TAU

fractions = []

for realization in ProgressBar(range(REALIZATIONS)):

    # Emit photons
    x = np.repeat(xs, N)
    y = np.repeat(ys, N)
    z = np.repeat(zs, N)

    # Sample wavelengths from spectrum
    xi = np.random.random(N)
    wav = np.interp(xi, cdf, lam)

    # Sample isotropic direction
    cost = np.random.uniform(-1, 1, N)
    sint = np.sqrt(1 - cost * cost)
    phi = np.random.uniform(0, 2 * np.pi, N)

    # Convert to directional vector
    vx = np.cos(phi) * sint
    vy = np.sin(phi) * sint
    vz = cost

    # Determine distance to intersection of ray with slab - the first intersection,
    # if any, will be with the top of the slab
    t = (zmax - z) / vz

    # Find new position
    x += vx * t
    y += vy * t
    z += vz * t

    # Check whether the intersection is inside the bounds for the slab
    intersect = (t > 0) & (x <= xmax) & (x >= xmin) & (y <= ymax) & (y >= ymin)

    # Keep only photons that intersect slab
    wav = wav[intersect]
    x = x[intersect]
    y = y[intersect]
    z = z[intersect]
    vx = vx[intersect]
    vy = vy[intersect]
    vz = vz[intersect]

    # Now search for the exit point for any of the photons that did intersect

    t1 = (zmin - z) / vz

    t2 = np.zeros_like(x)
    posvx = vx > 0
    t2[posvx] = (xmax - x[posvx]) / vx[posvx]
    t2[~posvx] = (xmin - x[~posvx]) / vx[~posvx]

    t3 = np.zeros_like(y)
    posvy = vy > 0
    t3[posvy] = (ymax - y[posvy]) / vy[posvy]
    t3[~posvy] = (ymin - y[~posvy]) / vy[~posvy]

    tnext = np.minimum(np.minimum(t1, t2), t3)

    # We now have the distance travelled through the slab for all photons. We now
    # assume the photons all travel through and the medium is completely optically
    # thin, so then the energy absorbed by the dust (using the pathlength calculation)
    # is the following

    fractions.append(np.sum(tnext * np.interp(wav, dust_wav, tau_unit)) / N)

mean = np.mean(fractions)
error = np.std(fractions) / len(fractions)

print("Fraction of energy absorbed: {0} +/- {1:.5f}%".format(mean, error / mean * 100))
