"""
Parameter conversions
"""

import numpy as np

def convert_to_beta_parameters(parameters, remove=True):
    """
    Convert to parameters for standard beta distribution.

    Calls gwpopulation.conversions.mu_var_max_to_alpha_beta_max

    Parameters
    ==========
    parameters: dict
        The input parameters
    remove: bool
        Whether to list the added keys

    Returns
    =======
    converted: dict
        The dictionary of parameters with the new parameters added
    added_keys: list
        The keys added to the dictionary, only non-empty if `remove=True`
    """
    added_keys = list()
    converted = parameters.copy()

    def _convert(suffix, param):
        alpha = f"alpha_{param}{suffix}"
        beta = f"beta_{param}{suffix}"
        mu = f"mu_{param}{suffix}"
        sigma = f"sigma_{param}{suffix}"
        amax = f"amax{suffix}"

        if alpha not in parameters.keys() or beta not in parameters.keys():
            needed = True
        elif converted[alpha] is None or converted[beta] is None:
            needed = True
        else:
            needed = False
            done = True

        if needed:
            if mu in converted.keys() and sigma in converted.keys():
                done = True
                converted[alpha], converted[beta], _, = mu_var_max_to_alpha_beta_max(
                    parameters[mu], parameters[sigma], parameters[amax]
                )
                if remove:
                    added_keys.append(alpha)
                    added_keys.append(beta)
            else:
                done = False
        return done

    for param in ["chi", "rho"]:
        done = False
        for suffix in ["_1", "_2"]:
            _done = _convert(suffix, param)
            done = done or _done
        if not done:
            _ = _convert("", param)
    
    for param in ['chi_eff', 'chi_dif']:
        converted[f'xi_{param}'], converted[f'omega_{param}'], converted[f'{param}_skew'] = (
            mu_var_gamma_to_xi_omega_alpha(
                parameters[f'mu_{param}'], parameters[f'sigma_{param}'], parameters[f'gamma_{param}']))
        added_keys.append(f'xi_{param}')
        added_keys.append(f'omega_{param}')
        added_keys.append(f'{param}_skew')
        if f'{param}_min' in parameters.keys():
            converted[f'{param}_min_con'] = parameters[f'mu_{param}'] - parameters[f'{param}_min']
            added_keys.append(f'{param}_min_con')
        if f'{param}_max' in parameters.keys():
            converted[f'{param}_max_con'] = parameters[f'{param}_max'] - parameters[f'mu_{param}']
            added_keys.append(f'{param}_max_con')

    return converted, added_keys


def alpha_beta_max_to_mu_var_max(alpha, beta, amax):
    r"""
    Convert between parameters for beta distribution

    .. math::
        \mu &= a_\max \frac{\alpha}{\alpha + \beta}

        \sigma^2 &= a_\max^2 \frac{\alpha\beta}{(\alpha + \beta)^2 + (\alpha + \beta + 1)^2}

    Parameters
    ==========
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    amax: float
        The maximum value (:math:`a_\max`)

    Returns
    =======
    mu: float
        The mean (:math:`\mu`)
    var: float
        The variance (:math:`\sigma^2`)
    amax: float
        The maximum spin (:math:`a_\max`)
    """
    mu = alpha / (alpha + beta) * amax
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)) * amax ** 2
    return mu, var, amax


def mu_var_max_to_alpha_beta_max(mu, var, amax):
    r"""
    Convert between parameters for beta distribution

    .. math::
        \alpha &= \frac{\mu^2 (a_\max - \mu) - \mu \sigma^2}{a_\max\sigma^2}

        \beta &= \frac{\mu (a_\max - \mu)^2 - (a_\max - \mu) \sigma^2}{a_\max\sigma^2}

    Parameters
    ==========
    mu: float
        The mean (:math:`\mu`)
    var: float
        The variance (:math:`\sigma^2`)
    amax: float
        The maximum value (:math:`a_\max`)

    Returns
    =======
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)
    amax: float
        The maximum spin (:math:`a_\max`)
    """
    mu /= amax
    var /= amax ** 2
    alpha = (mu ** 2 * (1 - mu) - mu * var) / var
    beta = (mu * (1 - mu) ** 2 - (1 - mu) * var) / var
    return alpha, beta, amax


def mu_var_gamma_to_xi_omega_alpha(mu, var, gamma):
    """
    Convert between parameters for skew norm distribution
    
    Parameters
    ==========
    mu : float
        The mean
    var : float
        The variance
    gamma : float
        The skewness
        
    Returns
    =======
    xi : float
        Shape parameter
    omega : float
        Shape parameter
    alpha : float
        Shape parameter
    """
    delta = np.sqrt(np.pi/(2 * (
        1 + ((4 - np.pi) / (2*np.abs(gamma)))**(2/3))))
    delta *= np.sign(gamma)
    alpha = delta / np.sqrt(1 - delta**2)
    omega = np.sqrt(var / (1 - 2*delta**2/np.pi))
    xi = mu - omega*delta*np.sqrt(2/np.pi)
    return xi, omega, alpha