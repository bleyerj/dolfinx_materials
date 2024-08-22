import numpy as np
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD
from .elasticity import LinearElasticIsotropic
import jax.numpy as jnp
import jax


class LinearViscoElasticity(JAXMaterial):

    def __init__(self, branch0, branch1, tau, nud):
        """_summary_

        Parameters
        ----------
        branch0 : LinearElastic
            First elastic branch
        branch1 : LinearElastic
            Second elastic branch, parallel to the dashpot
        tau : float
            Relaxation time
        nud : float
            Poisson ratio of the corresponding dashpot model
        """
        super().__init__()
        self.branch0 = branch0
        self.branch1 = branch1
        self.tau = tau
        self.nud = nud

    @property
    def internal_state_variables(self):
        return {"epsv": 6}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        epsv_old = state["epsv"]
        eps_old = state["Strain"]
        deps = eps - eps_old

        epsv_new = (
            eps
            + jnp.exp(-dt / self.tau) * (epsv_old - eps_old)
            - jnp.exp(-dt / 2 / self.tau) * deps
        )

        sig = self.branch0.C @ eps + self.branch1.C @ (eps - epsv_new)
        state["epsv"] = epsv_new
        state["Strain"] = eps
        state["Stress"] = sig
        return sig, state


class GeneralizedMaxwell(JAXMaterial):

    def __init__(
        self,
        bulk_modulus,
        shear_modulus,
        viscoelastic_bulk_modulus,
        viscoelastic_shear_modulus,
        relaxation_time,
    ):
        """A generalized N-branch Maxwell model.

        Parameters
        ----------
        bulk_modulus : float
            Elastic bulk modulus
        shear_modulus : float
            Elastic shear modulus
        viscoelastic_bulk_modulus : float, list, ndarray
            List of viscoelastic bulk moduli
        viscoelastic_shear_modulus : float, list, ndarray
            List of viscoelastic shear moduli
        relaxation_time : float, list, ndarray
            List of relaxation times
        """
        super().__init__()
        self.bulk_modulus = bulk_modulus
        self.shear_modulus = shear_modulus
        self.elastic_branch = LinearElasticIsotropic(
            kappa=bulk_modulus, mu=shear_modulus
        )
        self.viscoelastic_bulk_modulus = jnp.atleast_1d(viscoelastic_bulk_modulus)
        self.viscoelastic_shear_modulus = jnp.atleast_1d(viscoelastic_shear_modulus)
        self.relaxation_time = jnp.atleast_1d(relaxation_time)
        self.N_branch = len(self.relaxation_time)
        assert (
            len(self.viscoelastic_bulk_modulus) == self.N_branch
        ), "Number of viscoelastic bulk modulus should match number of relaxation times"
        assert (
            len(self.viscoelastic_shear_modulus) == self.N_branch
        ), "Number of viscoelastic shear modulus should match number of relaxation times"
        self.viscous_branch = [
            LinearElasticIsotropic(kappa=kappai, mu=mui)
            for kappai, mui in zip(
                self.viscoelastic_bulk_modulus, self.viscoelastic_shear_modulus
            )
        ]

    @property
    def internal_state_variables(self):
        return {"ViscoElasticStress": 6 * self.N_branch}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        sigv_old = jnp.reshape(state["ViscoElasticStress"], (self.N_branch, 6))
        eps_old = state["Strain"]
        deps = eps - eps_old

        sig_el = self.elastic_branch.C @ eps
        sigv = []
        for i in range(self.N_branch):
            a = jnp.exp(-dt / self.relaxation_time[i])
            b = self.relaxation_time[i] / dt * (1 - a)
            sigv1 = sigv_old[i] * a
            sigv2 = b * self.viscous_branch[i].C @ deps
            sigv.append(sigv1 + sigv2)

        sig = sig_el + sum(sigv)
        state["ViscoElasticStress"] = jnp.array(sigv).flatten()
        state["Strain"] = eps
        state["Stress"] = sig
        return sig, state
