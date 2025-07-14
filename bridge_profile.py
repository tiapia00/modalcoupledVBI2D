import numpy as np
from scipy.interpolate import CubicSpline

def generate_harmonic_profile(length, nx, base_freq, base_amp, num_harmonics=2, decay_exp=6.0, seed=None):
    """
    Generate a synthetic road profile with a base spatial frequency and decaying harmonics.

    Parameters:
    - length: total length of the road [m]
    - nx: number of spatial points
    - base_freq: base spatial frequency [1/m]
    - num_harmonics: number of harmonics to include
    - decay_exp: decay exponent for harmonics (amplitude ~ 1/n^decay_exp)
    - seed: for reproducibility

    Returns:
    - interp1d function for r(x)
    - x (spatial points)
    - r (road profile values)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, length, nx)
    r = np.zeros_like(x)

    for n in range(1, num_harmonics + 1):
        freq = n * base_freq  # nth harmonic
        amplitude = base_amp / (n ** decay_exp)
        r += amplitude * np.sin(2 * np.pi * freq * x)

    # Interpolating function
    r_interp = CubicSpline(x, r)

    return r_interp
