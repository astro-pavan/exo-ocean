import numpy as np

sigma = 5.670374419e-8
c = 299792458
h = 6.62607015e-34
k_B = 1.380649e-23

PI4_15 = np.pi**4 / 15
_B = {0: 1.0, 1: -0.5, 2: 1/6, 4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66,
      12: -691/2730, 14: 7/6, 16: -3617/510, 18: 43867/798, 20: -174611/330}


def _head(x):
    """integral_0^x t^3/(e^t-1) dt via Bernoulli series (use for x <= 1)."""
    acc = np.zeros_like(x)
    fact = 1.0
    for n in range(21):
        if n > 0:
            fact *= n
        if n in _B:
            acc += _B[n] * x**(n + 3) / (fact * (n + 3))
    return acc


def _tail(x, n_terms=50):
    """integral_x^inf t^3/(e^t-1) dt via exponential series (use for x > 1)."""
    k = np.arange(1, n_terms + 1).reshape((-1,) + (1,) * x.ndim)
    return np.sum(np.exp(-k * x) * (x**3 / k + 3 * x**2 / k**2
                                    + 6 * x / k**3 + 6 / k**4), axis=0)


def tail(x):
    """integral_x^inf t^3/(e^t-1) dt for any x >= 0 (including inf), vectorised."""
    x = np.asarray(x, dtype=float)
    small = x <= 1.0
    big = np.isinf(x)
    t = _tail(np.where(small | big, 2.0, x))
    t = np.where(big, 0.0, t)
    return np.where(small, PI4_15 - _head(np.where(small, x, 0.0)), t)


def band_integral(a, b):
    """integral_a^b t^3/(e^t-1) dt, a, b >= 0, vectorised and broadcastable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    both_small = (a <= 1.0) & (b <= 1.0)
    direct = _head(np.minimum(b, 1.0)) - _head(np.minimum(a, 1.0))
    return np.where(both_small, direct, tail(a) - tail(b))

def planck_function_integrated(T, nu1, nu2):

    u1 = h * nu1 / (k_B * T)
    u2 = h * nu2 / (k_B * T)

    f = band_integral(u1, u2)

    return sigma * T ** 4 * f / PI4_15