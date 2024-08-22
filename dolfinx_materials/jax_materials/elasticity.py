#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   22/08/2024
"""
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial
from .tensors import J, K


class LinearElasticIsotropic(JAXMaterial):
    def __init__(self, E=None, nu=None, kappa=None, mu=None):
        """A linear elastic isotropic material.

        Parameters
        ----------
        E : float, optional
            Young modulus, by default None
        nu : float, optional
            Poisson ratio, by default None
        kappa : float, optional
            Bulk modulus, by default None
        mu : float, optional
            Shear modulus, by default None

        Either (E, nu) or (kappa, mu) must be provided.
        """
        super().__init__()
        # Given (E, nu)
        if E is not None and nu is not None:
            kappa = E / (3 * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))

        # Given (kappa, mu)
        elif kappa is not None and mu is not None:
            E = 9 * kappa * mu / (3 * kappa + mu)
            nu = (3 * kappa - 2 * mu) / (2 * (3 * kappa + mu))
        else:
            raise ValueError(
                "Invalid combination of inputs. Provide either (E, nu) or (kappa, mu)."
            )

        self.E = E
        self.nu = nu
        self.kappa = kappa
        self.mu = mu
        self.C = 3 * self.kappa * J + 2 * self.mu * K

    def get_Lame_parameters(self, E, nu):
        return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)

    def compute_C(self, E, nu):
        lmbda, mu = self.get_Lame_parameters(E, nu)
        kappa = lmbda + 2 / 3 * mu
        return 3 * kappa * J + 2 * mu * K

    def constitutive_update(self, eps, state, dt):
        sig = jnp.dot(self.C, eps)
        state["Stress"] = sig
        return self.C, state


class PlaneStressLinearElasticIsotropic(LinearElasticIsotropic):

    def get_Lame_parameters(self, E, nu):
        lmbda_ = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        return 2 * mu * lmbda_ / (lmbda_ + 2 * mu), mu
