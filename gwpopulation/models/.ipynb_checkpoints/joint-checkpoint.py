from warnings import warn

from ..cupy_utils import trapz, cumtrapz, xp, betainc
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
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
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
        xi_chi_eff, omega_chi_eff: float
            Shape parameters of chi_eff distribution.
        chi_eff_min: float
            Minimum allowed chi_eff.
        chi_eff_max: float
            Maximum allowed chi_eff.
        chi_eff_skew: float
            Skewness of chi_eff Gaussian.
        kappa_q_chi_eff: float
            Correlation between chi_eff and mass ratio.
        xi_chi_dif, omega_chi_dif: float
            Shape parameters of chi_dif distribution.
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
                xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax)
                
        prob = p_mass*p_spin
        return prob
    
    def p_spin(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax):
        p_spin = self.p_chi_eff(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                                kappa_q_chi_eff)
        p_spin *= self.p_chi_dif(dataset, xi_chi_dif, omega_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew)
        p_spin *= self.p_rho(dataset, alpha_rho, beta_rho, amax)
        return p_spin
    
    def p_chi_eff(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                  xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                  kappa_q_chi_eff):
        u, v, chi_eff_norm = self.copula_coords(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                                xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew)
        p_chi_eff = truncskewnorm(dataset["chi_eff"], xi=xi_chi_eff, omega=omega_chi_eff,
                                  high=chi_eff_max, low=chi_eff_min, skew=chi_eff_skew)
        p_chi_eff /= chi_eff_norm
        p_chi_eff *= self.copula_function(u, v, kappa_q_chi_eff)
        return p_chi_eff
        
    def p_chi_dif(self, dataset, xi_chi_dif, omega_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew):
        p_chi_dif_grid = truncskewnorm(self.chi_effs, xi=xi_chi_dif, omega=omega_chi_dif,
                                       high=chi_dif_max, low=chi_dif_min, skew=chi_dif_skew)
        norm_chi_dif = trapz(p_chi_dif_grid, self.chi_effs)
        p_chi_dif = truncskewnorm(dataset["chi_dif"], xi=xi_chi_dif, omega=omega_chi_dif,
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
                      xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew):
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
        p_chi_eff = truncskewnorm(self.chi_effs, xi=xi_chi_eff, omega=omega_chi_eff,
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


class SPSMDEffectiveCopulaNormBase(SinglePeakSmoothedMassDistribution):
    """
    PL+P mass model with correlated spin model. This being a skewed gaussian chi_eff
    and chi_dif, with identical beta distributions for rho_1 and rho_2 and a
    copula correlating mass ratio and chi_eff.
    """
    def __init__(self, mmin=2, mmax=100):
        super(SPSMDEffectiveCopulaNormBase, self).__init__(mmin, mmax)
        self.chi_effs = xp.linspace(-1, 1, 500)
        self.rhos = xp.linspace(0, 1, 250)
        self.n_norm = 10000
    
    def __call__(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                 xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                 xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
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
        xi_chi_eff, omega_chi_eff: float
            Shape parameters of chi_eff distribution.
        chi_eff_min: float
            Minimum allowed chi_eff.
        chi_eff_max: float
            Maximum allowed chi_eff.
        chi_eff_skew: float
            Skewness of chi_eff Gaussian.
        kappa_q_chi_eff: float
            Correlation between chi_eff and mass ratio.
        xi_chi_dif, omega_chi_dif: float
            Shape parameters of chi_dif distribution.
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
        p_mass = super(SPSMDEffectiveCopulaNormBase, self).__call__(
            dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m)
        
        p_spin, cdf_q, cdf_chi_eff, cdf_chi_dif, cdf_rho = self.p_spin(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax)
        
        p_norm = self.p_norm(cdf_q, cdf_chi_eff, cdf_chi_dif, cdf_rho, kappa_q_chi_eff)
        
        prob = p_mass*p_spin/p_norm
        return prob
    
    def p_spin(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                xi_chi_eff, omega_chi_eff, chi_eff_min, chi_eff_max, chi_eff_skew, kappa_q_chi_eff, 
                xi_chi_dif, omega_chi_dif, chi_dif_min, chi_dif_max, chi_dif_skew,
                alpha_rho, beta_rho, amax):
        p_spin, cdf_q, cdf_chi_eff = self.p_chi_eff(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                                    xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                                                    kappa_q_chi_eff)
        p_dif, cdf_chi_dif = self.p_chi_dif(dataset, xi_chi_dif, omega_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew)
        p_spin *= p_dif
        p_rh, cdf_rho = self.p_rho(dataset, alpha_rho, beta_rho, amax)
        p_spin *= p_rh
        return p_spin, cdf_q, cdf_chi_eff, cdf_chi_dif, cdf_rho
    
    def p_chi_eff(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                  xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew,
                  kappa_q_chi_eff):
        u, v, chi_eff_norm, u_g, v_g = self.copula_coords(dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                                                          xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew)
        p_chi_eff = truncskewnorm(dataset["chi_eff"], xi=xi_chi_eff, omega=omega_chi_eff,
                                  high=chi_eff_max, low=chi_eff_min, skew=chi_eff_skew)
        p_chi_eff /= chi_eff_norm
        p_chi_eff *= self.copula_function(u, v, kappa_q_chi_eff)
        return p_chi_eff, u_g, v_g
        
    def p_chi_dif(self, dataset, xi_chi_dif, omega_chi_dif, chi_dif_max, chi_dif_min, chi_dif_skew):
        p_chi_dif_grid = truncskewnorm(self.chi_effs, xi=xi_chi_dif, omega=omega_chi_dif,
                                       high=chi_dif_max, low=chi_dif_min, skew=chi_dif_skew)
        cdf_chi_dif = cumtrapz(p_chi_dif_grid, self.chi_effs, initial=0)
        norm_chi_dif = xp.max(cdf_chi_dif)
        
        p_chi_dif = truncskewnorm(dataset["chi_dif"], xi=xi_chi_dif, omega=omega_chi_dif,
                                  high=chi_dif_max, low=chi_dif_min, skew=chi_dif_skew)
        p_chi_dif /= norm_chi_dif
        cdf_chi_dif /= norm_chi_dif
        return xp.nan_to_num(p_chi_dif), xp.nan_to_num(cdf_chi_dif)
    
    def p_rho(self, dataset, alpha_rho, beta_rho, amax):
        if alpha_rho < 0 or beta_rho < 0:
            return 0
        p_rho = beta_dist(dataset["rho_1"], alpha=alpha_rho, beta=beta_rho, scale=amax)
        p_rho *= beta_dist(dataset["rho_2"], alpha=alpha_rho, beta=beta_rho, scale=amax)
        cdf_rho = betainc(alpha_rho, beta_rho, self.rhos)
        return p_rho, xp.nan_to_num(cdf_rho)
    
    def copula_function(self, u, v, kappa):
        raise NotImplementedError
    
    def copula_coords(self, dataset, alpha, beta, mmin, mmax, lam, mpp, sigpp, delta_m,
                      xi_chi_eff, omega_chi_eff, chi_eff_max, chi_eff_min, chi_eff_skew):
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
        p_chi_eff = truncskewnorm(self.chi_effs, xi=xi_chi_eff, omega=omega_chi_eff,
                                  high=chi_eff_max, low=chi_eff_min, skew=chi_eff_skew)
        
        # v(chi_eff) grid
        v = cumtrapz(p_chi_eff, self.chi_effs, initial=0)
        chi_eff_norm = xp.max(v)
        v /= chi_eff_norm
        v = xp.nan_to_num(v)
        
        # Interpolate for v(chi_eff)
        res_v = xp.interp(dataset["chi_eff"], self.chi_effs, v)
        
        return res_u, res_v, chi_eff_norm, u, v
    
    def p_norm(cdf_q, cdf_chi_eff, cdf_chi_dif, cdf_rho, kappa):
        q_interp = xp.random.rand(self.n_norm)
        q_draws = xp.interp(q_interp, cdf_q, self.qs)
        chi_eff_draws = self.copula_draw(q_interp, kappa)
        chi_dif_draws = xp.interp(xp.random.rand(self.n_norm),
                                  cdf_chi_dif,
                                  self.chi_effs)
        rho1_draws = xp.interp(xp.random.rand(self.n_norm),
                               cdf_rho,
                               self.rhos)
        rho2_draws = xp.interp(xp.random.rand(self.n_norm),
                               cdf_rho,
                               self.rhos)
        
        chi1_draws, chi2_draws = self.chi_conversion(chi_eff_draws,
                                                     chi_dif_draws,
                                                     rho1_draws, rho2_draws,
                                                     q_draws)
        
        physical = (chi1_draws <= 1) & (chi2_draws <= 1)
        n_physical = len(chi1_draws[(physical)])
        norm = n_physical/self.n_norm
        
        return norm
        
    def chi_conversion(self, chi_eff, chi_dif, rho1, rho2, q):
        junk1 = (1 + q)*(q*chi_dif + chi_eff)/(1 + q**2)
        junk2 = (1 + q)*(q*chi_eff - chi_dif)/(1 + q**2)

        chi1 = xp.sqrt(junk1**2 + rho1**2)
        chi2 = xp.sqrt(junk2**2 + rho2**2)

        return xp.nan_to_num(chi1), xp.nan_to_num(chi2)
    
    def copula_draw(self, q_interp, kappa):
        raise NotImplementedError


class SPSMDEffectiveFrankCopulaNorm(SPSMDEffectiveCopulaNormBase):
    """
    SPSMDEffectiveCopulaBase model with Frank copula density.
    """
    def copula_function(self, u, v, kappa):
        return frank_copula(u, v, kappa)
    
    def copula_draw(self, u, kappa):
        v_interp = xp.random.rand(self.n_norm)
        v = (1/kappa * xp.log(1 + v_interp*(xp.exp(kappa) - 1)/
                 (v_interp - xp.exp(kappa*u) * (v_interp - 1))))
        return v
        