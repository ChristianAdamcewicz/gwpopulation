from warnings import warn

from ..cupy_utils import trapz, cumtrapz, xp
from ..utils import powerlaw, truncskewnorm, beta_dist, frank_copula, gaussian_copula, fgm_copula
from .mass import SinglePeakSmoothedMassDistribution, two_component_single


class SPSMDEffectiveCopulaBase(SinglePeakSmoothedMassDistribution):
    """
    PL+P mass model with correlated spin model. This being a skewed gaussian chi_eff
    and chi_dif, with identical beta distributions for rho_1 and rho_2 and a
    copula correlating mass ratio and chi_eff.
    """
    def __init__(self, mmin=2, mmax=100):
        super(SPSMDEffectiveCopulaBase, self).__init__(mmin, mmax)
        self.chi_effs = xp.linspace(-1, 1, 500)
    
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 mu_chi_eff, log_sigma_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 mu_chi_dif, log_sigma_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                 alpha_rho, beta_rho, amax,
                 lambda_chi_peak=0):
        """
        Parameters
        ----------
        dataset: dict
            Dictionary of numpy arrays for 'mass_1', 'mass_ratio', and 'chi_eff'.
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
        mu_chi_eff: float
            Mean of chi_eff Gaussian.
        log_sigma_chi_eff: float
            Log_10 of standard deviation for chi_eff Gaussian.
        chi_eff_min: float
            Minimum allowed chi_eff.
        chi_eff_max: float
            Maximum allowed chi_eff.
        chi_eff_skew: float
            Skewness of chi_eff Gaussian.
        kappa_q_chi_eff: float
            Correlation between chi_eff and mass ratio.
        mu_chi_dif: float
            Mean of chi_dif Gaussian.
        log_sigma_dif_eff: float
            Log_10 of standard deviation for chi_dif Gaussian.
        chi_dif_min: float
            Minimum allowed chi_dif.
        chi_dif_max: float
            Maximum allowed chi_dif.
        chi_dif_skew: float
            Skewness of chi_dif Gaussian.
        alpha_rho, beta_rho: float
            Parameters of rho distribution for both black holes.
        amax: float
            Maximum allowed spin magnitude.
        """ 
        p_mass = super(SPSMDEffectiveCopulaBase, self).__call__(
            dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m)
        
        p_spin = self.p_spin(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                mu_chi_eff, 10**log_sigma_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                mu_chi_dif, 10**log_sigma_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax)
                
        prob = p_mass*p_spin
        return prob
    
    def p_spin(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                mu_chi_eff, sigma_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                mu_chi_dif, sigma_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax):
        p_spin = self.p_chi_eff(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                mu_chi_eff, sigma_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                                kappa_q_chi_eff)
        p_spin *= self.p_chi_dif(dataset, mu_chi_dif, sigma_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew)
        p_spin *= self.p_rho(dataset, alpha_rho, beta_rho, amax)
        return p_spin
    
    def p_chi_eff(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                  mu_chi_eff, sigma_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                  kappa_q_chi_eff):
        u, v, chi_eff_norm = self.copula_coords(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                                mu_chi_eff, sigma_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew)
        p_chi_eff = truncskewnorm(dataset["chi_eff"], mu=mu_chi_eff, sigma=sigma_chi_eff,
                                  high=chi_eff_max, low=chi_eff_min, skew=chi_eff_skew)
        p_chi_eff /= chi_eff_norm
        p_chi_eff *= self.copula_function(u, v, kappa_q_chi_eff)
        return p_chi_eff
        
    def p_chi_dif(self, dataset, mu_chi_dif, sigma_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew):
        p_chi_dif_grid = truncskewnorm(self.chi_effs, mu=mu_chi_dif, sigma=sigma_chi_dif,
                                       high=chi_dif_max, low=chi_dif_min, skew=chi_dif_skew)
        norm_chi_dif = trapz(p_chi_dif_grid, self.chi_effs)
        p_chi_dif = truncskewnorm(dataset["chi_dif"], mu=mu_chi_dif, sigma=sigma_chi_dif,
                                  high=chi_dif_max, low=chi_dif_min, skew=chi_dif_skew)
        p_chi_dif /= norm_chi_dif
        return xp.nan_to_num(p_chi_dif)
    
    def p_rho(self, dataset, alpha_rho, beta_rho, amax):
        if alpha_rho < 0 or beta_rho < 0:
            return 0
        p_rho = beta_dist(dataset["rho_1"], alpha=alpha_rho, beta=beta_rho, scale=amax)
        p_rho *= beta_dist(dataset["rho_2"], alpha=alpha_rho, beta=beta_rho, scale=amax)
        return p_rho
    
    def copula_function(self, u, v, kappa):
        raise NotImplementedError
    
    def copula_coords(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                      mu_chi_eff, sigma_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew):
        '''Get u(q)'''
        # p(m1) grid
        p_m = two_component_single(
            self.m1s, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=100, delta_m=delta_m)
        p_m_norm = trapz(p_m, self.m1s)
        p_m /= p_m_norm
        p_m = xp.nan_to_num(p_m)

        # p(q|m1) grid
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m)
        p_q_norm = trapz(p_q, self.qs, axis=0)
        p_q /= p_q_norm
        p_q = xp.nan_to_num(p_q)

        # p(q) grid
        integrand_q_m = p_q * p_m
        p_q_marg = trapz(integrand_q_m, self.m1s, axis=-1)
        p_q_marg = xp.nan_to_num(p_q_marg)

        # u(q) grid
        u = cumtrapz(p_q_marg, self.qs, initial=0)
        u /= xp.max(u)
        u = xp.nan_to_num(u)

        # Interpolate for u(q)
        res_u = xp.interp(dataset["mass_ratio"], self.qs, u)
        
        '''get v(chi_eff)'''
        # p(chi_eff) grid
        p_chi_eff = truncskewnorm(self.chi_effs, mu=mu_chi_eff, sigma=sigma_chi_eff,
                                  high=chi_eff_max, low=chi_eff_min, skew=chi_eff_skew)
        
        # v(chi_eff) grid
        v = cumtrapz(p_chi_eff, self.chi_effs, initial=0)
        chi_eff_norm = xp.max(v)
        v /= chi_eff_norm
        v = xp.nan_to_num(v)
        
        # Interpolate for v(chi_eff)
        res_v = xp.interp(dataset["chi_eff"], self.chi_effs, v)
        
        return res_u, res_v, chi_eff_norm


class SPSMDEffectiveFrankCopula(SPSMDEffectiveCopulaBase):
    """
    SPSMDEffectiveCopulaBase model with Frank copula density.
    """
    def copula_function(self, u, v, kappa):
        return frank_copula(u, v, kappa)

    
class SPSMDEffectiveGaussianCopula(SPSMDEffectiveCopulaBase):
    """
    SPSMDEffectiveCopulaBase model with Gaussian copula density.
    """
    def copula_function(self, u, v, kappa):
        return gaussian_copula(u, v, kappa)
    
    
class SPSMDEffectiveFGMCopula(SPSMDEffectiveCopulaBase):
    """
    SPSMDEffectiveCopulaBase model with FGM copula density.
    """
    def copula_function(self, u, v, kappa):
        return fgm_copula(u, v, kappa)
