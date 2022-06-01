"""
Implemented mass models
"""

from warnings import warn
import inspect

from ..cupy_utils import trapz, xp
from ..utils import powerlaw, truncnorm


def matter_matters(mass, A, NSmin, NSmax, BHmin, BHmax, 
                   n0, n1, n2, n3, mbreak, alpha_1, alpha_2):
    r"""
    the single-mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178

    .. math::
        p(m|\lambda) = n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A) \times
            l(m|m_{\text{max}}, \eta) \\
                 \times \begin{cases}
                         & m^{\alpha_1} \text{ if } m < \gamma_{\text{low}} \\
                         & m^{\alpha_2} \text{ if } m > \gamma_{\text{low}} \\
                         & 0 \text{ otherwise }
                 \end{cases}.
    
    where $l(m|m_{\text{max}}, \eta)$ is the low pass filter with powerlaw $\eta$
    applied at mass $m_{\text{max}}$,
    $n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A)$ is the notch
    filter with depth $A$ applied between $\gamma_{\text{low}}$ and 
    $\gamma_{\text{high}}$, and
    $\lambda$ is the subset of hyperparameters $\{ \gamma_{\text{low}},
    \gamma_{\text{high}}, A, \alpha_1, \alpha_2, m_{\text{min}}, m_{\
    text{max}}\}$.
    
    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`). 
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`). 
    A: float
        depth of the dip between NSmax and BHmin (A).
    """
    mbreak=NSmax
    logprob = xp.where((mass >= 0.5)*(mass <= 350), 
                       -xp.log(1 + (NSmin/mass)**n0) \
                       + xp.log(1.0 - A/((1 + (NSmax/mass)**n1) * (1 + (mass/BHmin)**n2))) \
                       - xp.log(1 + (mass/BHmax)**n3)\
                       + xp.where(mass <= mbreak, 
                                  alpha_1,
                                  alpha_2
                                  )*(xp.log(mass) - xp.log(mbreak)),
                       -xp.infty)
    return xp.exp(logprob)

def double_power_law_primary_mass(mass, alpha_1, alpha_2, mmin, mmax, break_fraction):
    r"""
    Broken power-law mass distribution

    .. math::
        p(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for more massive black hole below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for more massive black hole above break (:math:`\alpha_2`).
    break_fraction: float
        The fraction between mmin and mmax primary mass distribution breaks (:math:`\delta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    """

    prob = xp.zeros_like(mass)
    m_break = mmin + break_fraction * (mmax - mmin)
    correction = powerlaw(m_break, alpha=-alpha_2, low=m_break, high=mmax) / powerlaw(
        m_break, alpha=-alpha_1, low=mmin, high=m_break
    )
    low_part = powerlaw(mass[mass < m_break], alpha=-alpha_1, low=mmin, high=m_break)
    prob[mass < m_break] = low_part * correction
    high_part = powerlaw(mass[mass >= m_break], alpha=-alpha_2, low=m_break, high=mmax)
    prob[mass >= m_break] = high_part
    return prob / (1 + correction)


def double_power_law_peak_primary_mass(
    mass, alpha_1, alpha_2, mmin, mmax, break_fraction, lam, mpp, sigpp
):
    r"""
    Broken power-law with a Gaussian component.

    .. math::
        p(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta, \lambda_m, \mu_m, \sigma_m) =
        (1 - \lambda_m) p_{\text{bpl}}(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta)
        + \lambda_m p_{\text{norm}}(m | \mu_m, \sigma_m)

    .. math::
        p_{\text{bpl}}(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    .. math::
        p_{\text{norm}}(m | \mu_m, \sigma_m) \propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for more massive black hole below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for more massive black hole above break (:math:`\alpha_2`).
    break_fraction:float
        The fraction between mmin and mmax primary mass distribution breaks (:math:`\delta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation fo the Gaussian component (:math:`\sigma_m`).
    """

    p_pow = double_power_law_primary_mass(
        mass=mass,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        mmin=mmin,
        mmax=mmax,
        break_fraction=break_fraction,
    )
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob

def matter_matters_primary_secondary_independent(dataset, A, NSmin, NSmax,
    BHmin, BHmax, n0, n1, n2, n3, mbreak, alpha_1, alpha_2
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as following independent distributions.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 
        'mass_ratio' q (:math:`m_2=m_1*q`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`). 
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`). 
    A: float
        depth of the dip between NSmax and BHmin (A).
    """

    p_m1 = matter_matters(dataset["mass_1"], A, NSmin, NSmax, BHmin, BHmax, 
                          n0, n1, n2, n3, mbreak, alpha_1, alpha_2)
    p_m2 = matter_matters(dataset["mass_2"], A, NSmin, 
                          NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak, 
                          alpha_1, alpha_2)
    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def matter_matters_pairing(dataset, A, NSmin, NSmax,
    BHmin, BHmax, n0, n1, n2, n3, mbreak, alpha_1, alpha_2, beta_q
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as being related by a pairing function.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 
        'mass_ratio' q (:math:`m_2=m_1*q`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    mbreak: float
        Mass at which the power law exponent switches from alpha_1 to alpha_2.
        Pinned for now to be at BHmin (:math:`\m_{break}`). 
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`). 
    A: float
        depth of the dip between NSmax and BHmin (A).
    beta_q: float
        power law governing mass ratio pairing function.
    """

    p_m1 = matter_matters(dataset["mass_1"], A, NSmin, NSmax, BHmin, BHmax, 
                          n0, n1, n2, n3, mbreak, alpha_1, alpha_2)
    p_m2 = matter_matters(dataset["mass_2"], A, NSmin, 
                          NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak, 
                          alpha_1, alpha_2)
    prob = _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_q, NSmin)
    return prob

def matter_matters_pairing_binned(dataset, A, NSmin, NSmax,
    BHmin, BHmax, n0, n1, n2, n3, mbreak_1, mbreak_2, alpha_1, alpha_2,
    beta_q_1, beta_q_2, beta_q_3, nbins,
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as being related by a pairing function.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 
        'mass_ratio' q (:math:`m_2=m_1*q`).
    alpha_1: float
        Powerlaw exponent for compact object below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact object above break (:math:`\alpha_2`).
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts (:math:`\gamma_{low}`)
    BHmin: float
        Mass at which the notch filter ends (:math:`\gamma_{high}`)
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    n{0,1,2,3}: float
        Exponents to set the sharpness of the low mass cutoff, low edge of dip,
        high edge of dip, and high mass cutoff, respectively (:math:`\eta_i`). 
    A: float
        depth of the dip between NSmax and BHmin (A).
    beta_q_1: float
        power law governing mass ratio pairing function at m2<mbreak_1
    beta_q_2: float
        power law governing mass ratio pairing function at mbreak_1<m2<mbreak_2
    beta_q_3: float 
        power law governing mass ratio pairing function at m2>mbreak_2
    mbreak_1: float
        location of upper edge of lowest m2 bin governing the pairing function.
        See beta_q_1 and beta_q_2.
    mbreak_2: float
        location of upper edge of middle m2 bin giverning the pairing function.
        See beta_q_2 and beta_q_3.
    nbins: int
        number of m2 bins for the pairing function. options:2 or 3. If 2,
        beta_q_3 and mbreak_2 are irrelevant.
    """

    p_m1 = matter_matters(dataset["mass_1"], A, NSmin, NSmax, BHmin, BHmax, 
                          n0, n1, n2, n3, mbreak_1, alpha_1, alpha_2)
    p_m2 = matter_matters(dataset["mass_2"], A, NSmin, 
                          NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak_1,
                          alpha_1, alpha_2)
    prob = _primary_secondary_mass_binned_pairing(dataset, p_m1, p_m2, beta_q_1,
    beta_q_2, beta_q_3, mbreak_1, mbreak_2, nbins, NSmin)
    return prob

def matter_matters_pairing_bpl(dataset, A, NSmin, NSmax,
    BHmin, BHmax, n0, n1, n2, n3, mbreak, alpha_1, alpha_2,
    beta_q_1, beta_q_2, qbreak):

    p_m1 = matter_matters(dataset["mass_1"], A, NSmin, NSmax, BHmin, BHmax, 
                          n0, n1, n2, n3, mbreak, alpha_1, alpha_2)
    p_m2 = matter_matters(dataset["mass_2"], A, NSmin, 
                          NSmax, BHmin, BHmax, n0, n1, n2, n3, mbreak,
                          alpha_1, alpha_2)
    
    def pairing(q,b1,b2,qbreak):
        b = xp.where(q < qbreak, b1, b2)
        return q**b

    prob =  _primary_secondary_general(dataset, p_m1, p_m2) \
    * pairing(dataset["mass_2"]/dataset["mass_1"],beta_q_1,beta_q_2,qbreak)
    return prob

def double_power_law_primary_power_law_mass_ratio(
    dataset, alpha_1, alpha_2, beta, mmin, mmax, break_fraction
):
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) = p_{\text{bpl}}(m_1) p(q | m_1)

    .. math::
        p_{\text{bpl}}(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    .. math::
        p(q | m_1) \propto q^\beta : \frac{m_1}{m_\min} \leq q \leq 1

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for `mass_1` (:math:`m_1`) and `mass_ratio` (:math:`q`).
    alpha_1: float
        Negative power law exponent for more massive black hole before break (:math:`\alpha_1`).
    alpha_2: float
        Negative power law exponent for more massive black hole after break (:math:`\alpha_2`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    break_fraction: float
        Break point of the primary mass distribution.
        This is specified as a fraction of the way between mmin and mmax.
        E.g., mmin=5, mmax=45, break_fraction=0.5 would have a break at 25
    beta: float
        Power law exponent of the mass ratio distribution.
    """
    params = dict(mmin=mmin, mmax=mmax, break_fraction=break_fraction)
    p_m1 = double_power_law_primary_mass(
        dataset["mass_1"], alpha_1=alpha_1, alpha_2=alpha_2, **params
    )
    p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
    prob = p_m1 * p_q
    return prob


def power_law_primary_mass_ratio(dataset, alpha, beta, mmin, mmax):
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) &= p_{\text{pow}}(m_1) p(q | m_1)

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p(q | m_1) &\propto q^\beta : \frac{m_1}{m_\min} \leq q \leq 1

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_ratio' (:math:`q`).
    alpha: float
        Negative power law exponent for more massive black hole (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    beta: float
        Power law exponent of the mass ratio distribution (:math:`\beta`).
    """
    return two_component_primary_mass_ratio(
        dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, lam=0, mpp=35, sigpp=1
    )


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset["mass_1"] >= dataset["mass_2"]) * 2

def _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_pair, min_m2):
    q = dataset["mass_2"]/dataset["mass_1"]
    pairing = powerlaw(q, beta_pair, 1, min_m2 / dataset["mass_1"])
    return _primary_secondary_general(dataset, p_m1, p_m2) * pairing

def _primary_secondary_mass_binned_pairing(dataset, p_m1, p_m2, beta_pair_1,
beta_pair_2, beta_pair_3, mbreak, mbreak_2, nbins):
    q = dataset["mass_2"]/dataset["mass_1"]
    min_m2=0.5
    max_m1=350
    if nbins == 2:
        beta_pair = xp.where(dataset["mass_2"] < mbreak, beta_pair_1, beta_pair_2)
        lower_q_bound = xp.where(dataset["mass_2"] < mbreak,
            min_m2/max_m1, mbreak/max_m1)
    elif nbins == 3:
        beta_pair = xp.where(dataset["mass_2"] < mbreak, beta_pair_1, 
                             xp.where(dataset["mass_2"] < mbreak_2, beta_pair_2,
                                      beta_pair_3)
                             )
        lower_q_bound = xp.where(dataset["mass_2"] < mbreak,
                                 min_m2/max_m1, 
                                 xp.where(dataset["mass_2"] < mbreak_2, 
                                          mbreak/max_m1,
                                          mbreak_2/max_m1
                                         )
                                 )
    else:
        print("only 2 or 3 bin options are supported!")
    pairing = powerlaw(q, beta_pair, 1, lower_q_bound)
    return _primary_secondary_general(dataset, p_m1, p_m2) * (q ** beta_pair)


def power_law_primary_secondary_independent(dataset, alpha, beta, mmin, mmax):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m1, m2) &= p_{\text{pow}}(m1) p_{\text{pow}}(m2) : m1 \geq m2

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_2' (:math:`m_2`).
    alpha: float
        Negative power law exponent for more massive black hole (:math:`\alpha`).
    beta: float
        Negative power law exponent of the secondary mass distribution (:math:`\beta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    """
    p_m1 = powerlaw(dataset["mass_1"], -alpha, mmax, mmin)
    p_m2 = powerlaw(dataset["mass_2"], -beta, mmax, mmin)
    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def power_law_primary_secondary_identical(dataset, alpha, mmin, mmax):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2 | \alpha, m_\min, m_\max) &= p_{\text{pow}}(m_1 | \alpha) p_{\text{pow}}(m_2 | \alpha) : m_1 \geq m_2

        p_{\text{pow}}(m | \alpha) &\propto m^{-\alpha} : m_\min \leq m < m_\max

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' (:math:`m_1`) and 'mass_2' (:math:`m_2`).
    alpha: float
        Negative power law exponent for both black holes (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    """
    return power_law_primary_secondary_independent(
        dataset=dataset, alpha=alpha, beta=alpha, mmin=mmin, mmax=mmax
    )


def two_component_single(mass, alpha, mmin, mmax, lam, mpp, sigpp):
    r"""
    Power law model for one-dimensional mass distribution with a Gaussian component.

    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}} + \lambda_m p_{\text{norm}}

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob


def three_component_single(
    mass, alpha, mmin, mmax, lam, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2
):
    r"""
    Power law model for one-dimensional mass distribution with two Gaussian components.

    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}}(m) + \lambda_m p_{\text{norm}}(m)

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Array of mass values.
    alpha: float
        Negative power law exponent for the black hole distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    """
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm1 = truncnorm(mass, mu=mpp_1, sigma=sigpp_1, high=100, low=mmin)
    p_norm2 = truncnorm(mass, mu=mpp_2, sigma=sigpp_2, high=100, low=mmin)
    prob = (1 - lam) * p_pow + lam * lam_1 * p_norm1 + lam * (1 - lam_1) * p_norm2
    return prob


def two_component_primary_mass_ratio(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp):
    r"""
    Power law model for two-dimensional mass distribution, modelling primary
    mass and conditional mass ratio distribution.

    .. math::
        p(m_1, q) = p(m1) p(q | m_1)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    beta: float
        Power law exponent of the mass ratio distribution.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    params = dict(mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
    prob = p_m1 * p_q
    return prob


def two_component_primary_secondary_independent(
    dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp
):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2) = p_{\text{pow}}(m_1) p_{\text{pow}}(m_2) : m1 \geq m_2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    beta: float
        Negative power law exponent of the secondary mass distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    params = dict(mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
    p_m1 = two_component_single(dataset["mass_1"], alpha=alpha, **params)
    p_m2 = two_component_single(dataset["mass_2"], alpha=beta, **params)

    prob = _primary_secondary_general(dataset, p_m1, p_m2)
    return prob


def two_component_primary_secondary_identical(
    dataset, alpha, mmin, mmax, lam, mpp, sigpp
):
    r"""
    Power law model for two-dimensional mass distribution, modelling the
    primary and secondary masses as following independent distributions.

    .. math::
        p(m_1, m_2) = p(m_1) * p(m_2) : m_1 \geq m_2

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_2'.
    alpha: float
        Negative power law exponent for more massive black hole.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum black hole mass.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation fo the Gaussian component.
    """
    return two_component_primary_secondary_independent(
        dataset=dataset,
        alpha=alpha,
        beta=alpha,
        mmin=mmin,
        mmax=mmax,
        lam=lam,
        mpp=mpp,
        sigpp=sigpp,
    )

class _PairingMassDistribution(object):
    """
    Generic mass distribution with a pairing function base class.
    
    Implements mass distributions of the form:
    
    .. math::
        p(m_1, m_2) = p_m(m_1) * p_m(m_2) * f_p(q) : m_1 \geq m_2
    
    """
    def __init__(self, mmin=0.5, mmax=350.):
        self.mmin = mmin
        self.mmax = mmax
        self.qmin = mmin/mmax
    
    def __call__(self, dataset, **kwargs):
        raise NotImplementedError

    def p_m(self, *args, **kwargs):
        raise NotImplementedError

    def pairing(self, *args, **kwargs):
        raise NotImplementedError

    def p_m1_m2(self, dataset, **kwargs):
        # parse arguments for use in pm(m) vs fp(q)
        pm_args = [k for k, v in 
                   inspect.signature(self.p_m).parameters.items()
                   if k not in ["self","mass"]
                  ]
        pm_dict = {k: kwargs[k] for k in dict(kwargs) if k in pm_args}
        
        fp_args = [k for k, v in 
                   inspect.signature(self.pairing).parameters.items()
                   if k not in ["self","dataset"]
                  ]
        fp_dict = {k: kwargs[k] for k in dict(kwargs) if k in fp_args}
        
        p_m1 = xp.where((dataset['mass_1']>=self.mmin)*(dataset['mass_1']<=self.mmax),
                        self.p_m(dataset['mass_1'], **pm_dict),
                        0.0
                        )
        p_m2 = xp.where((dataset['mass_2']>=self.mmin)*(dataset['mass_2']<=self.mmax),
                        self.p_m(dataset['mass_2'], **pm_dict),
                        0.0
                        )
        fp = self.pairing(dataset, **fp_dict)

        return _primary_secondary_general(dataset, p_m1, p_m2) * fp

class _NotchFilterPairingMassDistribution(_PairingMassDistribution):
    """
    Generic pairing function mass distribution with "matter matters" 1D mass
    model base class.
    """
    def p_m(self, mass, A, NSmin, NSmax, BHmin, BHmax, 
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2):
        return matter_matters(mass, A, NSmin, NSmax, BHmin, BHmax,
        n0, n1, n2, n3, mbreak, alpha_1, alpha_2)

class NotchFilterPowerLawPairingMassDistribution(_NotchFilterPairingMassDistribution):
    """
    "Power Law + Dip + Break" Model.
    """
    def pairing(self, dataset, beta_q):
        mass_ratio = dataset["mass_2"]/dataset["mass_1"]
        return powerlaw(mass_ratio, beta_q, 1, self.qmin)

    def __call__(self, dataset, A, NSmin, NSmax, BHmin, BHmax,
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2, beta_q
            ):
        # get arguments in a dict
        kwargs = locals()
        kwargs.pop('self')
        return self.p_m1_m2(**kwargs)

class NotchFilterBrokenPowerLawPairingMassDistribution(_NotchFilterPairingMassDistribution):
    """
    Pairing function is a broken powerlaw in mass ratio
    """
    def pairing(self, dataset, beta_q_1, beta_q_2, qbreak):
        mass_ratio = dataset["mass_2"]/dataset["mass_1"]
        prob = xp.zeros_like(mass_ratio)

        correction = powerlaw(qbreak, alpha=beta_q_2, low=qbreak, high=1.) / powerlaw(
            qbreak, alpha=beta_q_1, low=self.qmin, high=qbreak
        )
        low_part = powerlaw(mass_ratio[mass_ratio < q_break], alpha=beta_q_1,
                            low=self.qmin, high=qbreak
                           )
        high_part = powerlaw(mass_ratio[mass_ratio >= q_break], alpha=beta_q_2,
                             low=q_break, high=1.
                            )
        prob[mass_ratio < qbreak] = low_part * correction
        prob[mass_ratio >= qbreak] = high_part
        return prob / (1 + correction)
#        beta_pair = xp.where(mass_ratio < qbreak, beta_q_1, beta_q_2)
#
#        return powerlaw(mass_ratio, beta_pair, 1, self.qmin)
    
    def __call__(self, dataset, A, NSmin, NSmax, BHmin, BHmax,
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2, 
            beta_q_1, beta_q_2, qbreak
            ):
        # get arguments in a dict
        kwargs = locals()
        kwargs.pop('self')
        return self.p_m1_m2(**kwargs)

class NotchFilterTruncatedGaussianPairingMassDistribution(_NotchFilterPairingMassDistribution):
    """
    Pairing function is a gaussian in mass ratio, truncated at q=1 and
    q=m1/Nsmin
    """
    def pairing(self, dataset, sigq, meanq):
        mass_ratio = dataset["mass_2"]/dataset["mass_1"]
        return truncnorm(mass_ratio, mu=meanq, sigma=sigq, high=1.,
            low=self.qmin)

    def __call__(self, dataset, A, NSmin, NSmax, BHmin, BHmax,
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2, 
            sigq, meanq
            ):
        # get arguments in a dict
        kwargs = locals()
        kwargs.pop('self')
        return self.p_m1_m2(**kwargs)

class NotchFilterDoubleGaussianPairingMassDistribution(_NotchFilterPairingMassDistribution):
    """
    Pairing function is a mixture of two gaussians in mass ratio, 
    with mixing fraction `lamq` between the two gaussians.  
    Both Gaussians are truncated at q=1 and q=`mmax`/`Nsmin`.
    """
    def pairing(self, dataset, lamq, sigq_1, meanq_1, sigq_2, meanq_2):
        mass_ratio = dataset["mass_2"]/dataset["mass_1"]
        p_g1 = truncnorm(mass_ratio, mu=meanq_1, sigma=sigq_1, high=1.,low=self.qmin)
        p_g2 = truncnorm(mass_ratio, mu=meanq_2, sigma=sigq_2, high=1.,low=self.qmin)
        prob = (1 - lamq) * p_g1 + lamq * p_g2
        return prob

    def __call__(self, dataset, A, NSmin, NSmax, BHmin, BHmax,
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2, 
            lamq, sigq_1, meanq_1, sigq_2, meanq_2
            ):
        # get arguments in a dict
        kwargs = locals()
        kwargs.pop('self')
        return self.p_m1_m2(**kwargs)

class NotchFilterPowerLawGaussianPairingMassDistribution(_NotchFilterPairingMassDistribution):
    """
    Pairing function is a mixture of a powerlaw and a Gaussian in mass ratio, 
    with mixing fraction `lamq` between the two components.  
    Both components are truncated at q=1 and q=`mmax`/`Nsmin`.
    """
    def pairing(self, dataset, lamq, sigq, meanq, beta_q):
        mass_ratio = dataset["mass_2"]/dataset["mass_1"]
        p_g = truncnorm(mass_ratio, mu=meanq, sigma=sigq, high=1.,low=self.qmin)
        p_pow = powerlaw(mass_ratio, beta_q, 1, self.qmin)
        prob = (1 - lamq) * p_pow + lamq * p_g
        return prob

    def __call__(self, dataset, A, NSmin, NSmax, BHmin, BHmax,
            n0, n1, n2, n3, mbreak, alpha_1, alpha_2, 
            lamq, sigq, meanq, beta_q
            ):
        # get arguments in a dict
        kwargs = locals()
        kwargs.pop('self')
        return self.p_m1_m2(**kwargs)

class _SmoothedMassDistribution(object):
    """
    Generic smoothed mass distribution base class.

    Implements the low-mass smoothing and power-law mass ratio
    distribution. Requires p_m1 to be implemented.
    """

    def __init__(self):
        self.m1s = xp.linspace(2, 100, 1000)
        self.qs = xp.linspace(0.001, 1, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def p_m1(self, *args, **kwargs):
        raise NotImplementedError

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )
        try:
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        if delta_m == 0.0:
            return 1
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )
        norms = trapz(p_q, self.qs, axis=0)

        all_norms = (
            norms[self.n_below] * (1 - self.step) + norms[self.n_above] * self.step
        )

        return all_norms

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        self.n_below = xp.zeros_like(masses, dtype=xp.int) - 1
        m_below = xp.zeros_like(masses)
        for mm in self.m1s:
            self.n_below += masses > mm
            m_below[masses > mm] = mm
        self.n_above = self.n_below + 1
        max_idx = len(self.m1s)
        self.n_below[self.n_below < 0] = 0
        self.n_above[self.n_above == max_idx] = max_idx - 1
        self.step = xp.minimum((masses - m_below) / self.dm, 1)

    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
        """
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.

        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)

        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
        window = xp.ones_like(masses)
        if delta_m > 0.0:
            smoothing_region = (masses >= mmin) & (masses < (mmin + delta_m))
            shifted_mass = masses[smoothing_region] - mmin
            if shifted_mass.size:
                exponent = xp.nan_to_num(
                    delta_m / shifted_mass + delta_m / (shifted_mass - delta_m)
                )
                window[smoothing_region] = 1 / (xp.exp(exponent) + 1)
        window[(masses < mmin) | (masses > mmax)] = 0
        return window


class SinglePeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m):
        """
        Powerlaw + peak model for two-dimensional mass distribution with low
        mass smoothing.

        https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha: float
            Powerlaw exponent for more massive black hole.
        beta: float
            Power law exponent of the mass ratio distribution.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        mpp: float
            Mean of the Gaussian component.
        sigpp: float
            Standard deviation fo the Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.

        Notes
        -----
        The interpolation of the p(q) normalisation has a fill value of
        the normalisation factor for m_1 = 100.
        """
        p_m1 = self.p_m1(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        p_m = two_component_single(
            dataset["mass_1"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_m1(self, alpha, mmin, mmax, lam, mpp, sigpp, delta_m):
        """Calculate the normalisation factor for the primary mass"""
        if delta_m == 0.0:
            return 1
        p_m = two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)

        norm = trapz(p_m, self.m1s)
        return norm


class SmoothedMassDistribution(SinglePeakSmoothedMassDistribution):
    def __init__(self):
        warn(
            "SmoothedMassDistribution has been deprecated and will be removed in a "
            "future release, use SinglePeakSmoothedMassDistribution instead.",
            DeprecationWarning,
        )
        super(SmoothedMassDistribution, self).__init__()


class MultiPeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self,
        dataset,
        alpha,
        beta,
        mmin,
        mmax,
        lam,
        lam_1,
        mpp_1,
        sigpp_1,
        mpp_2,
        sigpp_2,
        delta_m,
    ):
        """
        Powerlaw + two peak model for two-dimensional mass distribution with
        low mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha: float
            Powerlaw exponent for more massive black hole.
        beta: float
            Power law exponent of the mass ratio distribution.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        lam_1: float
            Fraction of black holes in the lower mass Gaussian component.
        mpp_1: float
            Mean of the lower mass Gaussian component.
        mpp_2: float
            Mean of the upper mass Gaussian component.
        sigpp_1: float
            Standard deviation of the lower mass Gaussian component.
        sigpp_2: float
            Standard deviation of the upper mass Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
        """
        p_m1 = self.p_m1(
            dataset,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
            delta_m=delta_m,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(
        self,
        dataset,
        alpha,
        mmin,
        mmax,
        lam,
        lam_1,
        mpp_1,
        sigpp_1,
        mpp_2,
        sigpp_2,
        delta_m,
    ):
        p_m = three_component_single(
            dataset["mass_1"],
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
            delta_m=delta_m,
        )
        return p_m / norm

    def norm_p_m1(
        self, alpha, mmin, mmax, lam, lam_1, mpp_1, sigpp_1, mpp_2, sigpp_2, delta_m
    ):
        if delta_m == 0.0:
            return 1
        p_m = three_component_single(
            self.m1s,
            alpha=alpha,
            mmin=mmin,
            mmax=mmax,
            lam=lam,
            lam_1=lam_1,
            mpp_1=mpp_1,
            mpp_2=mpp_2,
            sigpp_1=sigpp_1,
            sigpp_2=sigpp_2,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm


class BrokenPowerLawSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self,
        dataset,
        alpha_1,
        alpha_2,
        beta,
        mmin,
        mmax,
        delta_m,
        break_fraction,
    ):
        """
        Broken power law for two-dimensional mass distribution with low
        mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha_1: float
            Powerlaw exponent for more massive black hole below break.
        alpha_2: float
            Powerlaw exponent for more massive black hole above break.
        beta: float
            Power law exponent of the mass ratio distribution.
        break_fraction: float
            Fraction between mmin and mmax primary mass distribution breaks at.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        mpp: float
            Mean of the Gaussian component.
        sigpp: float
            Standard deviation fo the Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
        """

        p_m1 = self.p_m1(
            dataset,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction):
        p_m = double_power_law_primary_mass(
            dataset["mass_1"],
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
        )
        return p_m / norm

    def norm_p_m1(self, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction):
        if delta_m == 0.0:
            return 1
        p_m = double_power_law_primary_mass(
            self.m1s,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm


class BrokenPowerLawPeakSmoothedMassDistribution(_SmoothedMassDistribution):
    def __call__(
        self,
        dataset,
        alpha_1,
        alpha_2,
        beta,
        mmin,
        mmax,
        delta_m,
        break_fraction,
        lam,
        mpp,
        sigpp,
    ):
        """
        Broken power law for two-dimensional mass distribution with low
        mass smoothing.

        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
        alpha_1: float
            Powerlaw exponent for more massive black hole below break.
        alpha_2: float
            Powerlaw exponent for more massive black hole above break.
        beta: float
            Power law exponent of the mass ratio distribution.
        break_fraction: float
            Fraction between mmin and mmax primary mass distribution breaks at.
        mmin: float
            Minimum black hole mass.
        mmax: float
            Maximum mass in the powerlaw distributed component.
        lam: float
            Fraction of black holes in the Gaussian component.
        mpp: float
            Mean of the Gaussian component.
        sigpp: float
            Standard deviation fo the Gaussian component.
        delta_m: float
            Rise length of the low end of the mass distribution.
        """

        p_m1 = self.p_m1(
            dataset,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(
        self,
        dataset,
        alpha_1,
        alpha_2,
        mmin,
        mmax,
        delta_m,
        break_fraction,
        lam,
        mpp,
        sigpp,
    ):
        p_m = double_power_law_peak_primary_mass(
            dataset["mass_1"],
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(dataset["mass_1"], mmin=mmin, mmax=100, delta_m=delta_m)
        norm = self.norm_p_m1(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            delta_m=delta_m,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        return p_m / norm

    def norm_p_m1(
        self, alpha_1, alpha_2, mmin, mmax, delta_m, break_fraction, lam, mpp, sigpp
    ):
        if delta_m == 0.0:
            return 1
        p_m = double_power_law_peak_primary_mass(
            self.m1s,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            mmin=mmin,
            mmax=mmax,
            break_fraction=break_fraction,
            lam=lam,
            mpp=mpp,
            sigpp=sigpp,
        )
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        norm = trapz(p_m, self.m1s)
        return norm
