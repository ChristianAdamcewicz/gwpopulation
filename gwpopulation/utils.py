"""
Helper functions for probability distributions.
"""

import os

from .cupy_utils import erf, betaln, xp, erfinv


def beta_dist(xx, alpha, beta, scale=1):
    r"""
    Beta distribution probability

    .. math::
        p(x) = \frac{x^{\alpha - 1} (x_\max - x)^{\beta - 1}}{B(\alpha, \beta) x_\max^{\alpha + \beta + 1}}

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    scale: float, array-like
        A scale factor for the distribution of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    if alpha < 0:
        raise ValueError(f"Parameter alpha must be greater or equal zero, low={alpha}.")
    if beta < 0:
        raise ValueError(f"Parameter beta must be greater or equal zero, low={beta}.")
    ln_beta = (alpha - 1) * xp.log(xx) + (beta - 1) * xp.log(scale - xx)
    ln_beta -= betaln(alpha, beta)
    ln_beta -= (alpha + beta - 1) * xp.log(scale)
    prob = xp.exp(ln_beta)
    prob = xp.nan_to_num(prob)
    prob *= (xx >= 0) * (xx <= scale)
    return prob


def powerlaw(xx, alpha, high, low):
    r"""
    Power-law probability

    .. math::
        p(x) = \frac{1 + \alpha}{x_\max^{1 + \alpha} - x_\min^{1 + \alpha}} x^\alpha

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    alpha: float, array-like
        The spectral index of the distribution (:math:`\alpha`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    if xp.any(xp.asarray(low) < 0):
        raise ValueError(f"Parameter low must be greater or equal zero, low={low}.")
    if alpha == -1:
        norm = 1 / xp.log(high / low)
    else:
        norm = (1 + alpha) / (high ** (1 + alpha) - low ** (1 + alpha))
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncnorm(xx, mu, sigma, high, low):
    r"""
    Truncated normal probability

    .. math::
        p(x) =
        \sqrt{\frac{2}{\pi\sigma^2}}
        \left[\text{erf}\left(\frac{x_\max - \mu}{\sqrt{2}}\right) + \text{erf}\left(\frac{\mu - x_\min}{\sqrt{2}}\right)\right]^{-1}
        \exp\left(-\frac{(\mu - x)^2}{2 \sigma^2}\right)

    Parameters
    ----------
    xx: float, array-like
        The abscissa values (:math:`x`)
    mu: float, array-like
        The mean of the normal distribution (:math:`\mu`)
    sigma: float
        The standard deviation of the distribution (:math:`\sigma`)
    high: float, array-like
        The maximum of the distribution (:math:`x_\min`)
    low: float, array-like
        The minimum of the distribution (:math:`x_\max`)

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at `xx`

    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0, sigma={sigma}")
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncskewnorm(xx, xi, omega, high, low, skew):
    if omega <= 0:
        raise ValueError(f"omega must be greater than 0, omega={omega}")
    prob = xp.exp(-xp.power(xx - xi, 2) / (2 * omega ** 2))
    prob *= (1 + erf(skew * (xx - xi) / (xp.sqrt(2) * omega)))
    prob *= (xx <= high) & (xx >= low)
    return xp.nan_to_num(prob)


def frank_copula(u, v, kappa):
    """
    Frank copula density function.
    
    Parameters
    ----------
    u: float, array-like
        CDF of first parameter.
    v: float, array-like
        CDF of second parameter.
    kappa: float
        Level of correlation

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at (u,v)
    """
    if kappa == 0:
        return 1.
    expkap = xp.exp(kappa)
    expkapuv = expkap**(u + v)
    prob = kappa * expkapuv * (expkap - 1) / (expkap - expkap**u - expkap**v + expkapuv)**2
    return xp.nan_to_num(prob)


def gaussian_copula(u, v, kappa):
    """
    Gaussian copula density function.
    
    Parameters
    ----------
    u: float, array-like
        CDF of first parameter.
    v: float, array-like
        CDF of second parameter.
    kappa: float
        Level of correlation

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at (u,v)
    """
    if kappa < -1 or kappa > 1:
        raise ValueError(f"kappa must be in range (-1,1), kappa={kappa}")
    if kappa == 0:
        return 1.
    a = xp.sqrt(2) * erfinv(2*u - 1.)
    b = xp.sqrt(2) * erfinv(2*v - 1.)
    kappa2 = kappa**2
    prob = xp.exp(-((a**2 + b**2)*kappa2 - 2*a*b*kappa) / (2*(1. - kappa2))) / xp.sqrt(1. - kappa2)
    return xp.nan_to_num(prob)


def fgm_copula(u, v, kappa):
    """
    Frank copula density function.
    
    Parameters
    ----------
    u: float, array-like
        CDF of first parameter.
    v: float, array-like
        CDF of second parameter.
    kappa: float
        Level of correlation

    Returns
    -------
    prob: float, array-like
        The distribution evaluated at (u,v)
    """
    if kappa < -1 or kappa > 1:
        raise ValueError(f"kappa must be in range (-1,1), kappa={kappa}")
    if kappa == 0:
        return 1.
    prob = 1. + kappa*(1. - 2*u)*(1. - 2*v)
    return prob


def unnormalized_2d_gaussian(xx, yy, mu_x, mu_y, sigma_x, sigma_y, covariance):
    determinant = sigma_x ** 2 * sigma_y ** 2 * (1 - covariance)
    residual_x = (mu_x - xx) * sigma_x
    residual_y = (mu_y - yy) * sigma_y
    prob = xp.exp(
        -(residual_x ** 2 + residual_y ** 2 - 2 * residual_x * residual_y * covariance)
        / 2
        / determinant
    )
    return prob


def effective_jacobian(q, a1, a2, z1, z2):
    """
    Computes the Jacobian:
    |d(chi_eff, chi_dif, rho_1, rho_2)/d(a_1, a_2, z_1, z_2)|^-1
    
    Parameters
    ----------
    q: float, array-like
        Mass ratio value (according to the convention q<1)
    a1: float, array-like
        Spin magnitude of heavier BH
    a2: float, array-like
        Spin magnitude of lighter BH
    z1: float, array-like
        Spin orientation (cos[t]) of heavier BH
    z2: float, array-like
        Spin orientation (cos[t]) of lighter BH

    Returns
    -------
    result: float, array-like
        Jacobian
    """
    import numpy as np
    log_numerator = np.nan_to_num(
        2*np.log(1.+q) + 0.5*(np.log(1.-z1**2) + np.log(1.-z2**2)))
    log_denominator = np.nan_to_num(
        np.log(1.+q**2) + np.log(a1) + np.log(a2))
    log_result = log_numerator - log_denominator
    result = np.exp(log_result)
    return np.nan_to_num(result)


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
