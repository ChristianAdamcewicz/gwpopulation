"""
Implemented joint models
"""
import inspect

from ..cupy_utils import trapz, cumtrapz, betainc, xp
from ..utils import frank_copula, powerlaw, beta_dist, truncnorm

from .mass import (
    BaseSmoothedMassDistribution,
    BaseSmoothedComponentMassDistribution,
    two_component_single,
    three_component_single,
)
from .spin import EffectiveSpin


class MassRatioChiEffCopulaBase(BaseSmoothedMassDistribution, EffectiveSpin):
    """
    """

    primary_model = None

    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m"]
        vars += self.spin_keys
        vars += ["kappa_q_chi_eff"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500)):
        BaseSmoothedMassDistribution.__init__(self, mmin, mmax, normalization_shape)
        EffectiveSpin.__init__(self)
        self.spin_keys = ["xi_eff", "omega_eff", "skew_eff",
                          "xi_diff", "omega_diff", "skew_diff",
                          "alpha_rho", "beta_rho"]

    def __call__(self, dataset, *args, **kwargs):
        spin_kwargs = dict()
        for key in self.spin_keys:
            spin_kwargs[key] = kwargs.pop(key)
        kappa_q_chi_eff = kwargs.pop("kappa_q_chi_eff")
        prob = BaseSmoothedMassDistribution.__call__(self, dataset, *args, **kwargs)
        prob *= EffectiveSpin.__call__(self, dataset, **spin_kwargs)
        prob *= self.copula(dataset, kappa_q_chi_eff)
        return prob
    
    def copula(self, dataset, kappa_q_chi_eff):
        u = xp.interp(dataset["mass_ratio"], self.qs, self.u)
        v = xp.interp(dataset["chi_eff"], self.chi_eff_diff, self.v)
        prob = frank_copula(u, v, kappa_q_chi_eff)
        return prob
    
    def norm_p_chi_eff(self, xi_eff, omega_eff, skew_eff):
        prob = self.p_chi_eff(self.chi_eff_diff, xi_eff, omega_eff, skew_eff)
        v = cumtrapz(prob, self.chi_eff_diff, initial=0)
        norm = xp.max(v)
        self.v = v/norm
        return norm

    def norm_p_m1(self, delta_m, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        if delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        self._p_m = p_m

        norm = trapz(p_m, self.m1s)
        return norm

    def norm_p_q(self, beta, mmin, delta_m):
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )
        norms = trapz(p_q, self.qs, axis=0)

        all_norms = (
            norms[self.n_below] * (1 - self.step) + norms[self.n_above] * self.step
        )

        p_q_marginalised = trapz(p_q*self._p_m, self.m1s, axis=1)
        u = cumtrapz(p_q_marginalised, self.qs, initial=0)
        self.u = u/xp.max(u)

        return all_norms


class MassRatioChiEffCopulaSPSMD(MassRatioChiEffCopulaBase):
    """
    """

    primary_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MassSingleSpinCopulaBase(BaseSmoothedComponentMassDistribution):
    """
    """

    mass_model = None

    @property
    def variable_names(self):
        vars = getattr(
            self.mass_model,
            "variable_names",
            inspect.getfullargspec(self.mass_model).args[1:],
        )
        vars += ["beta", "delta_m"]
        vars += self.spin_keys
        vars += self.kappa_keys
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __call__(self, dataset, *args, **kwargs):
        spin_kwargs = dict()
        kappa_kwargs = dict()
        for key in self.spin_keys:
            spin_kwargs[key] = kwargs.pop(key)
        for key in self.kappa_keys:
            kappa_kwargs[key] = kwargs.pop(key)
        prob = BaseSmoothedComponentMassDistribution.__call__(self, dataset, *args, **kwargs)
        prob *= self.spin_model(dataset, **spin_kwargs)
        prob *= self.copula(dataset, **kappa_kwargs, **spin_kwargs)
        return prob

    def norm(self, beta, **kwargs):
        p_m = self.p_m(self.norm_dataset, **kwargs)
        p_pairing = self.p_pairing(self.norm_dataset, beta=beta)
        prob = p_m*p_pairing
        norm = self.get_u_grid(prob)
        return norm


class MassPrimarySpinCopulaBase(MassSingleSpinCopulaBase):
    """
    """
    def __init__(self, mmin=2, mmax=100, normalization_shape=1000):
        self.spin_keys = ["alpha_chi", "beta_chi", "sigma_spin"]
        self.kappa_keys = ["kappa_s1"]
        MassSingleSpinCopulaBase.__init__(self, mmin, mmax, normalization_shape)
    
    def spin_model(self, dataset, alpha_chi, beta_chi, sigma_spin):
        p_chi1 = beta_dist(dataset["a_1"], alpha_chi, beta_chi, scale=1)
        p_tilt1 = truncnorm(dataset["cos_tilt_1"], 1, sigma_spin, 1, -1)
        return p_chi1 * p_tilt1
    
    def get_u_grid(self, p_m1_m2):
        p_m1 = trapz(p_m1_m2, self.ms, axis=0)
        u1 = cumtrapz(p_m1, self.ms, initial=0)
        norm = xp.max(u1)
        self.u1 = u1/norm
        return norm

    def get_v(self, dataset, alpha_chi, beta_chi):
        v1 = betainc(alpha_chi, beta_chi, dataset["a_1"])
        return v1

    def copula(self, dataset, kappa_s1, alpha_chi, beta_chi, sigma_spin):
        u1 = xp.interp(dataset["mass_1"], self.ms, self.u1)
        v1 = self.get_v(dataset, alpha_chi, beta_chi)
        copula1 = frank_copula(u1, v1, kappa_s1)
        return copula1
    

class MassPrimarySpinCopulaSPSMD(MassPrimarySpinCopulaBase):
    """
    """

    mass_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MassPrimarySpinCopulaMPSMD(MassPrimarySpinCopulaBase):
    """
    """

    mass_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MassSecondarySpinCopulaBase(MassSingleSpinCopulaBase):
    """
    """
    def __init__(self, mmin=2, mmax=100, normalization_shape=1000):
        self.spin_keys = ["alpha_chi", "beta_chi", "sigma_spin"]
        self.kappa_keys = ["kappa_s2"]
        MassSingleSpinCopulaBase.__init__(self, mmin, mmax, normalization_shape)
    
    def spin_model(self, dataset, alpha_chi, beta_chi, sigma_spin):
        p_chi2 = beta_dist(dataset["a_2"], alpha_chi, beta_chi, scale=1)
        p_tilt2 = truncnorm(dataset["cos_tilt_2"], 1, sigma_spin, 1, -1)
        return p_chi2 * p_tilt2
    
    def get_u_grid(self, p_m1_m2):
        p_m2 = trapz(p_m1_m2, self.ms, axis=1)
        u2 = cumtrapz(p_m2, self.ms, initial=0)
        norm = xp.max(u2)
        self.u2 = u2/norm
        return norm

    def get_v(self, dataset, alpha_chi, beta_chi):
        v2 = betainc(alpha_chi, beta_chi, dataset["a_2"])
        return v2

    def copula(self, dataset, kappa_s2, alpha_chi, beta_chi, sigma_spin):
        u2 = xp.interp(dataset["mass_2"], self.ms, self.u2)
        v2 = self.get_v(dataset, alpha_chi, beta_chi)
        copula2 = frank_copula(u2, v2, kappa_s2)
        return copula2
    

class MassSecondarySpinCopulaSPSMD(MassSecondarySpinCopulaBase):
    """
    """

    mass_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MassSecondarySpinCopulaMPSMD(MassSecondarySpinCopulaBase):
    """
    """

    mass_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class MassBothSpinCopulaBase(MassSingleSpinCopulaBase):
    """
    """
    def __init__(self, mmin=2, mmax=100, normalization_shape=1000):
        self.spin_keys = ["alpha_chi", "beta_chi", "sigma_spin"]
        self.kappa_keys = ["kappa_b1", "kappa_b2"]
        BaseSmoothedComponentMassDistribution.__init__(self, mmin, mmax, normalization_shape)
    
    def spin_model(self, dataset, alpha_chi, beta_chi, sigma_spin):
        p_chi1 = beta_dist(dataset["a_1"], alpha_chi, beta_chi, scale=1)
        p_chi2 = beta_dist(dataset["a_2"], alpha_chi, beta_chi, scale=1)
        p_tilt1 = truncnorm(dataset["cos_tilt_1"], 1, sigma_spin, 1, -1)
        p_tilt2 = truncnorm(dataset["cos_tilt_2"], 1, sigma_spin, 1, -1)
        return p_chi1 * p_chi2 * p_tilt1 * p_tilt2

    def get_u_grid(self, p_m1_m2):
        p_m1 = trapz(p_m1_m2, self.ms, axis=0)
        p_m2 = trapz(p_m1_m2, self.ms, axis=1)
        u1 = cumtrapz(p_m1, self.ms, initial=0)
        u2 = cumtrapz(p_m2, self.ms, initial=0)
        norm = xp.max(u1)
        self.u1 = u1/norm
        self.u2 = u2/norm
        return norm

    def get_v(self, dataset, alpha_chi, beta_chi):
        v1 = betainc(alpha_chi, beta_chi, dataset["a_1"])
        v2 = betainc(alpha_chi, beta_chi, dataset["a_2"])
        return v1, v2

    def copula(self, dataset, kappa_b1, kappa_b2, alpha_chi, beta_chi, sigma_spin):
        u1 = xp.interp(dataset["mass_1"], self.ms, self.u1)
        u2 = xp.interp(dataset["mass_2"], self.ms, self.u2)
        v1, v2 = self.get_v(dataset, alpha_chi, beta_chi)
        copula1 = frank_copula(u1, v1, kappa_b1)
        copula2 = frank_copula(u2, v2, kappa_b2)
        return copula1 * copula2


class MassBothSpinCopulaSPSMD(MassBothSpinCopulaBase):
    """
    """

    mass_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
    

class MassBothSpinCopulaMPSMD(MassBothSpinCopulaBase):
    """
    """

    mass_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)