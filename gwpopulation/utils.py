"""
Helper functions for probability distributions.
"""

import os

from .cupy_utils import erf, betaln, xp


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
    norm = 2 ** 0.5 / xp.pi ** 0.5 / sigma
    norm /= erf((high - mu) / 2 ** 0.5 / sigma) + erf((mu - low) / 2 ** 0.5 / sigma)
    prob = xp.exp(-xp.power(xx - mu, 2) / (2 * sigma ** 2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return xp.nan_to_num(prob)


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


def frank_copula(u, v, kappa):
    '''
    Frank copula density function.
    '''
    if kappa == 0:
        prob = 1.

    else:
        expkap = xp.exp(kappa)
        expkapuv = expkap**(u + v)

        prob = kappa * expkapuv * (expkap - 1) / (expkap - expkap**u - expkap**v + expkapuv)**2

    return prob


def polevl(x, coef, N):
    '''
    Used in PL function.
    '''
    y = 0.0

    for i in range(N+1):
        y += coef[-1-i]*x**i

    return y


def PL(x):
    '''
    Adapted from https://github.com/scipy/scipy/blob/main/scipy/special/cephes/spence.c

    Computes Spence’s function.
    '''
    A = xp.array(
        [4.65128586073990045278E-5,
         7.31589045238094711071E-3,
         1.33847639578309018650E-1,
         8.79691311754530315341E-1,
         2.71149851196553469920E0,
         4.25697156008121755724E0,
         3.29771340985225106936E0,
         1.00000000000000000126E0])
    B = xp.array(
        [6.90990488912553276999E-4,
         2.54043763932544379113E-2,
         2.82974860602568089943E-1,
         1.41172597751831069617E0,
         3.63800533345137075418E0,
         5.03278880143316990390E0,
         3.54771340985225096217E0,
         9.99999999999999998740E-1])

    xs = xp.copy(x)
    w = xp.zeros_like(xs)
    y = xp.zeros_like(xs)
    flag = xp.zeros_like(xs, dtype=int)

    xs[(xs > 2.0)] = 1.0 / xs[(xs > 2.0)]
    flag[(x > 2.0)] |= 2

    w[(xs > 1.5)] = (1.0 / xs[(xs > 1.5)]) - 1.0
    flag[(xs > 1.5)] |= 2

    w[(xs < 0.5)] = -xs[(xs < 0.5)]
    flag[(xs < 0.5)] |= 1

    w[((xs >= 0.5) & (xs <= 1.5))] = xs[((xs >= 0.5) & (xs <= 1.5))] - 1.0

    y = -w * polevl(w, A, 7) / polevl(w, B, 7)

    cond1 = ((flag >= 1) & (flag != 2))
    y[cond1] = (xp.pi*xp.pi) / 6.0 - xp.log(xs[cond1]) * xp.log(1.0 - xs[cond1]) - y[cond1]

    cond2 = (flag >= 2)
    y[cond2] = -0.5 * xp.log(xs[cond2])**2 - y[cond2]

    y[(x == 1.0)] = 0.0
    y[(x == 0.0)] = xp.pi*xp.pi/6.0

    return y


def Di(z):
    '''
    Used in chi_effective_prior_from_isotropic_spins function.
    '''
    return PL(1.-z+0j)


def chi_effective_prior_from_isotropic_spins(q,aMax,chi_eff):
    """
    Adapted from https://github.com/tcallister/effective-spin-priors/blob/main/priors.py

    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` and 'qs' is an array and take absolute value
    xs = xp.reshape(xp.abs(chi_eff),-1)
    qs = xp.reshape(q,-1)


    # Set up various piecewise cases
    pdfs = xp.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-qs)/(1.+qs))*(xs<qs*aMax/(1.+qs))
    caseB = (xs<aMax*(1.-qs)/(1.+qs))*(xs>qs*aMax/(1.+qs))
    caseC = (xs>aMax*(1.-qs)/(1.+qs))*(xs<qs*aMax/(1.+qs))
    caseD = (xs>aMax*(1.-qs)/(1.+qs))*(xs<aMax/(1.+qs))*(xs>=qs*aMax/(1.+qs))
    caseE = (xs>aMax*(1.-qs)/(1.+qs))*(xs>aMax/(1.+qs))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins and mass ratios
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    q_Z = qs[caseZ]
    q_A = qs[caseA]
    q_B = qs[caseB]
    q_C = qs[caseC]
    q_D = qs[caseD]
    q_E = qs[caseE]

    pdfs[caseZ] = (1.+q_Z)/(2.*aMax)*(2.-xp.log(q_Z))

    pdfs[caseA] = (1.+q_A)/(4.*q_A*aMax**2)*(
                    q_A*aMax*(4.+2.*xp.log(aMax) - xp.log(q_A**2*aMax**2 - (1.+q_A)**2*x_A**2))
                    - 2.*(1.+q_A)*x_A*xp.arctanh((1.+q_A)*x_A/(q_A*aMax))
                    + (1.+q_A)*x_A*(Di(-q_A*aMax/((1.+q_A)*x_A)) - Di(q_A*aMax/((1.+q_A)*x_A)))
                    )

    pdfs[caseB] = (1.+q_B)/(4.*q_B*aMax**2)*(
                    4.*q_B*aMax
                    + 2.*q_B*aMax*xp.log(aMax)
                    - 2.*(1.+q_B)*x_B*xp.arctanh(q_B*aMax/((1.+q_B)*x_B))
                    - q_B*aMax*xp.log((1.+q_B)**2*x_B**2 - q_B**2*aMax**2)
                    + (1.+q_B)*x_B*(Di(-q_B*aMax/((1.+q_B)*x_B)) - Di(q_B*aMax/((1.+q_B)*x_B)))
                    )

    pdfs[caseC] = (1.+q_C)/(4.*q_C*aMax**2)*(
                    2.*(1.+q_C)*(aMax-x_C)
                    - (1.+q_C)*x_C*xp.log(aMax)**2.
                    + (aMax + (1.+q_C)*x_C*xp.log((1.+q_C)*x_C))*xp.log(q_C*aMax/(aMax-(1.+q_C)*x_C))
                    - (1.+q_C)*x_C*xp.log(aMax)*(2. + xp.log(q_C) - xp.log(aMax-(1.+q_C)*x_C))
                    + q_C*aMax*xp.log(aMax/(q_C*aMax-(1.+q_C)*x_C))
                    + (1.+q_C)*x_C*xp.log((aMax-(1.+q_C)*x_C)*(q_C*aMax-(1.+q_C)*x_C)/q_C)
                    + (1.+q_C)*x_C*(Di(1.-aMax/((1.+q_C)*x_C)) - Di(q_C*aMax/((1.+q_C)*x_C)))
                    )

    pdfs[caseD] = (1.+q_D)/(4.*q_D*aMax**2)*(
                    -x_D*xp.log(aMax)**2
                    + 2.*(1.+q_D)*(aMax-x_D)
                    + q_D*aMax*xp.log(aMax/((1.+q_D)*x_D-q_D*aMax))
                    + aMax*xp.log(q_D*aMax/(aMax-(1.+q_D)*x_D))
                    - x_D*xp.log(aMax)*(2.*(1.+q_D) - xp.log((1.+q_D)*x_D) - q_D*xp.log((1.+q_D)*x_D/aMax))
                    + (1.+q_D)*x_D*xp.log((-q_D*aMax+(1.+q_D)*x_D)*(aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*xp.log(aMax/((1.+q_D)*x_D))*xp.log((aMax-(1.+q_D)*x_D)/q_D)
                    + (1.+q_D)*x_D*(Di(1.-aMax/((1.+q_D)*x_D)) - Di(q_D*aMax/((1.+q_D)*x_D)))
                    )

    pdfs[caseE] = (1.+q_E)/(4.*q_E*aMax**2)*(
                    2.*(1.+q_E)*(aMax-x_E)
                    - (1.+q_E)*x_E*xp.log(aMax)**2
                    + xp.log(aMax)*(
                        aMax
                        -2.*(1.+q_E)*x_E
                        -(1.+q_E)*x_E*xp.log(q_E/((1.+q_E)*x_E-aMax))
                        )
                    - aMax*xp.log(((1.+q_E)*x_E-aMax)/q_E)
                    + (1.+q_E)*x_E*xp.log(((1.+q_E)*x_E-aMax)*((1.+q_E)*x_E-q_E*aMax)/q_E)
                    + (1.+q_E)*x_E*xp.log((1.+q_E)*x_E)*xp.log(q_E*aMax/((1.+q_E)*x_E-aMax))
                    - q_E*aMax*xp.log(((1.+q_E)*x_E-q_E*aMax)/aMax)
                    + (1.+q_E)*x_E*(Di(1.-aMax/((1.+q_E)*x_E)) - Di(q_E*aMax/((1.+q_E)*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if xp.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q[boundary],aMax,xs[boundary]-1e-6))

    return xp.reshape(xp.real(pdfs), chi_eff.shape)


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
