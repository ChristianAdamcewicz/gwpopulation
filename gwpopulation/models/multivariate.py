"""
Implemented multivariate models.
"""

import inspect

import numpy as np
import scipy.integrate as sci

from ..utils import frank_copula, powerlaw
from .mass import (
    BaseSmoothedMassDistribution,
    two_component_single,
    three_component_single,
)
from .spin import EffectiveSpin
from .redshift import PowerLawRedshift

xp = np


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
        vars += ["kappa"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500)):
        BaseSmoothedMassDistribution.__init__(self, mmin, mmax, normalization_shape)
        EffectiveSpin.__init__(self)
        self.spin_keys = ["xi_eff", "omega_eff", "skew_eff",
                          "xi_diff", "omega_diff", "skew_diff",
                          "alpha_rho", "beta_rho"]

    def __call__(self, dataset, *args, **kwargs):
        spin_kwargs = {key:kwargs.pop(key) for key in self.spin_keys}
        kappa = kwargs.pop("kappa")
        prob = BaseSmoothedMassDistribution.__call__(self, dataset, *args, **kwargs)
        prob *= EffectiveSpin.__call__(self, dataset, **spin_kwargs)
        prob *= self.copula(dataset, kappa)
        return prob
    
    def copula(self, dataset, kappa):
        u = xp.interp(dataset["mass_ratio"], self.qs, self.u)
        v = xp.interp(dataset["chi_eff"], self.chi_eff_diff, self.v)
        prob = frank_copula(u, v, kappa)
        return prob
    
    def norm_p_chi_eff(self, xi_eff, omega_eff, skew_eff):
        prob = self.p_chi_eff(self.chi_eff_diff, xi_eff, omega_eff, skew_eff)
        v = sci.cumtrapz(prob, self.chi_eff_diff, initial=0)
        norm = xp.max(v)
        self.v = v/norm
        return norm

    def norm_p_m1(self, delta_m, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        self._p_m = p_m

        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

    def norm_p_q(self, beta, mmin, delta_m):
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )

        p_q_marginalised = xp.trapz(p_q*self._p_m, self.m1s, axis=1)
        u = sci.cumtrapz(p_q_marginalised, self.qs, initial=0)
        self.u = u/xp.max(u)

        norms = xp.nan_to_num(xp.trapz(p_q, self.qs, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )

        return self._q_interpolant(norms)


class MassRatioChiEffCopulaSPSMD(MassRatioChiEffCopulaBase):
    """
    """

    primary_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
    

class MassRatioChiEffCopulaMPSMD(MassRatioChiEffCopulaBase):
    """
    """

    primary_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class PrimaryMassChiEffCopulaBase(BaseSmoothedMassDistribution, EffectiveSpin):
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
        vars += ["kappa"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500)):
        BaseSmoothedMassDistribution.__init__(self, mmin, mmax, normalization_shape)
        EffectiveSpin.__init__(self)
        self.spin_keys = ["xi_eff", "omega_eff", "skew_eff",
                          "xi_diff", "omega_diff", "skew_diff",
                          "alpha_rho", "beta_rho"]

    def __call__(self, dataset, *args, **kwargs):
        spin_kwargs = {key:kwargs.pop(key) for key in self.spin_keys}
        kappa = kwargs.pop("kappa")
        prob = BaseSmoothedMassDistribution.__call__(self, dataset, *args, **kwargs)
        prob *= EffectiveSpin.__call__(self, dataset, **spin_kwargs)
        prob *= self.copula(dataset, kappa)
        return prob
    
    def copula(self, dataset, kappa):
        u = xp.interp(dataset["mass_1"], self.m1s, self.u)
        v = xp.interp(dataset["chi_eff"], self.chi_eff_diff, self.v)
        prob = frank_copula(u, v, kappa)
        return prob
    
    def norm_p_chi_eff(self, xi_eff, omega_eff, skew_eff):
        prob = self.p_chi_eff(self.chi_eff_diff, xi_eff, omega_eff, skew_eff)
        v = sci.cumtrapz(prob, self.chi_eff_diff, initial=0)
        norm = xp.max(v)
        self.v = v/norm
        return norm

    def norm_p_m1(self, delta_m, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        u = sci.cumtrapz(p_m, self.m1s, initial=0)
        self.u = u/xp.max(u)

        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm


class PrimaryMassChiEffCopulaSPSMD(PrimaryMassChiEffCopulaBase):
    """
    """

    primary_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
    BaseSmoothedMassDistribution


class PrimaryMassChiEffCopulaMPSMD(PrimaryMassChiEffCopulaBase):
    """
    """

    primary_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class RedshiftChiEffCopula(PowerLawRedshift, EffectiveSpin):
    """
    """
    @property
    def variable_names(self):
        vars = self.cosmology_names.copy()
        if self.base_variable_names is not None:
            vars += self.base_variable_names
        vars += self.spin_keys
        vars += ["kappa"]
        return vars

    def __init__(self, z_max=2.3, cosmo_model="Planck15"):
        PowerLawRedshift.__init__(self, z_max, cosmo_model)
        EffectiveSpin.__init__(self)
        self.spin_keys = ["xi_eff", "omega_eff", "skew_eff",
                          "xi_diff", "omega_diff", "skew_diff",
                          "alpha_rho", "beta_rho"]

    def __call__(self, dataset, *args, **kwargs):
        spin_kwargs = {key:kwargs.pop(key) for key in self.spin_keys}
        kappa = kwargs.pop("kappa")
        prob = PowerLawRedshift.__call__(self, dataset, **kwargs)
        prob *= EffectiveSpin.__call__(self, dataset, **spin_kwargs)
        prob *= self.copula(dataset, kappa)
        return prob
    
    def copula(self, dataset, kappa):
        u = xp.interp(dataset["redshift"], self.zs, self.u)
        v = xp.interp(dataset["chi_eff"], self.chi_eff_diff, self.v)
        prob = frank_copula(u, v, kappa)
        return prob
    
    def norm_p_chi_eff(self, xi_eff, omega_eff, skew_eff):
        prob = self.p_chi_eff(self.chi_eff_diff, xi_eff, omega_eff, skew_eff)
        v = sci.cumtrapz(prob, self.chi_eff_diff, initial=0)
        norm = xp.max(v)
        self.v = v/norm
        return norm

    def normalisation(self, parameters):
        normalisation_data = self.differential_spacetime_volume(
            dict(redshift=self.zs), bounds=True, **parameters
        )
        u = sci.cumtrapz(normalisation_data, self.zs, initial=0)
        norm = xp.max(u)
        self.u = u/norm
        return norm


class PrimaryMassRedshiftCopulaBase(BaseSmoothedMassDistribution, PowerLawRedshift):
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
        vars += self.cosmology_names.copy()
        if self.base_variable_names is not None:
            vars += self.base_variable_names
        vars += ["kappa"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500), z_max=2.3, cosmo_model="Planck15"):
        BaseSmoothedMassDistribution.__init__(self, mmin, mmax, normalization_shape)
        PowerLawRedshift.__init__(self, z_max, cosmo_model)

    def __call__(self, dataset, *args, **kwargs):
        redshift_kwargs = {key:kwargs.pop(key) for key in self.cosmology_names + self.base_variable_names}
        kappa = kwargs.pop("kappa")
        prob = BaseSmoothedMassDistribution.__call__(self, dataset, *args, **kwargs)
        prob *= PowerLawRedshift.__call__(self, dataset, **redshift_kwargs)
        prob *= self.copula(dataset, kappa)
        return prob
    
    def copula(self, dataset, kappa):
        u = xp.interp(dataset["mass_1"], self.m1s, self.u)
        v = xp.interp(dataset["redshift"], self.zs, self.v)
        prob = frank_copula(u, v, kappa)
        return prob

    def norm_p_m1(self, delta_m, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        u = sci.cumtrapz(p_m, self.m1s, initial=0)
        self.u = u/xp.max(u)

        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

    def normalisation(self, parameters):
        normalisation_data = self.differential_spacetime_volume(
            dict(redshift=self.zs), bounds=True, **parameters
        )
        v = sci.cumtrapz(normalisation_data, self.zs, initial=0)
        norm = xp.max(v)
        self.v = v/norm
        return norm


class PrimaryMassRedshiftCopulaSPSMD(PrimaryMassRedshiftCopulaBase):
    """
    """

    primary_model = two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class PrimaryMassRedshiftCopulaMPSMD(PrimaryMassRedshiftCopulaBase):
    """
    """

    primary_model = three_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
