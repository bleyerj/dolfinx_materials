import numpy as np
from .elasticity import LinearElasticIsotropic
from dolfinx_materials.material import Material
from dolfinx_materials.material.generic import tangent_AD
from dolfinx_materials.jax_newton_solver import JAXNewton
from .tensors import K
import jax
import jax.numpy as jnp


class ElastoPlasticIsotropicHardening(Material):
    def __init__(self, elastic_model, yield_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress

    @property
    def internal_state_variables(self):
        return {"p": 1}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # scalar instead of 1-dim vector
        sig_old = state["Stress"]
        jax.debug.print("Shape={s}", s=state["p"].shape)
        elastic_model = self.elastic_model
        E, nu = elastic_model.E, elastic_model.nu
        lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = jnp.sqrt(3 / 2.0) * jnp.linalg.norm(s_el)
        yield_criterion = sig_eq_el - sig_Y_old

        def r(dp):
            r_elastic = lambda dp: dp
            r_plastic = (
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)
            )
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)

        def deps_p(dp, yield_criterion):
            def deps_p_elastic(dp, yield_criterion):
                return jnp.zeros(6)

            def deps_p_plastic(dp, yield_criterion):
                return 3 / 2 * s_el / sig_eq_el * dp

            return jax.lax.cond(
                yield_criterion < 0.0,
                deps_p_elastic,
                deps_p_plastic,
                dp,
                yield_criterion,
            )

        # newton = JAXNewton(lambda x: jnp.array([r(x[0])]))
        # dp = 0.0
        # x = newton.solve(jnp.array([dp]))
        # dp = x[0]

        newton = JAXNewton(r)
        dp, res = newton.solve(0.0)

        sig = sig_el - 2 * mu * deps_p(dp, yield_criterion)

        # dR_dp = jax.grad(self.yield_stress)(p_old + dp)
        # n_el = s_el / sig_eq_el  # normal vector
        # beta = 3 * mu * dp / sig_eq_el
        # gamma = 3 * mu / (3 * mu + dR_dp)
        # D = 3 * mu * (gamma - beta) * jnp.outer(n_el, n_el) + 2 * mu * beta * K()

        # def Ct(dp, yield_criterion):
        #     def Ct_elastic(dp, yield_criterion):
        #         return C

        #     def Ct_plastic(dp, yield_criterion):
        #         return C - D

        #     return jax.lax.cond(
        #         yield_criterion < 0.0,
        #         Ct_elastic,
        #         Ct_plastic,
        #         dp,
        #         yield_criterion,
        #     )

        state["Strain"] = eps_old + deps
        state["p"] += dp
        state["Stress"] = sig
        return sig, state
