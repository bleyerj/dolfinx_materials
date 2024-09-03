import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton
from .tensors import dev, to_mat
from time import time


def von_Mises_stress(sig):
    return jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig))


def Hosford_stress(sig, a=10):
    sI = jnp.linalg.eigh(to_mat(sig))[0]
    return (
        1
        / 2
        * (
            jnp.abs(sI[0] - sI[1]) ** a
            + jnp.abs(sI[0] - sI[2]) ** a
            + jnp.abs(sI[2] - sI[1]) ** a
        )
    ) ** (1 / a)


class vonMisesIsotropicHardening(JAXMaterial):
    def __init__(self, elastic_model, yield_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_stress = von_Mises_stress
        self.newton_solver = JAXNewton()

    @property
    def internal_state_variables(self):
        return {"p": 1}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        sig_old = state["Stress"]

        mu = self.elastic_model.mu
        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = jnp.clip(self.equivalent_stress(sig_el), a_min=1e-8)
        n_el = dev(sig_el) / sig_eq_el
        yield_criterion = sig_eq_el - sig_Y_old

        def deps_p_elastic(dp):
            return jnp.zeros(6)

        def deps_p_plastic(dp):
            return 3 / 2 * n_el * dp  # n=n_el simplification due to radial return

        def deps_p(dp, yield_criterion):
            return jax.lax.cond(
                yield_criterion < 0.0,
                deps_p_elastic,
                deps_p_plastic,
                dp,
            )

        def r(dp):
            r_elastic = lambda dp: dp
            r_plastic = (
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)
            )
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)

        self.newton_solver.set_residual(r)
        dp, data = self.newton_solver.solve(0.0)

        sig = sig_el - 2 * mu * deps_p(dp, yield_criterion)

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        return sig, state


class GeneralIsotropicHardening(JAXMaterial):

    def __init__(self, elastic_model, yield_stress, equivalent_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = jax.jit(yield_stress)
        self.equivalent_stress = jax.jit(equivalent_stress)
        self.newton_solver = JAXNewton()

    @property
    def internal_state_variables(self):
        return {"p": 1}

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]

        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        sig_old = state["Stress"]

        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = jnp.clip(self.equivalent_stress(sig_el), a_min=1e-8)
        yield_criterion = sig_eq_el - sig_Y_old

        def stress(deps_p):
            return sig_old + C @ (deps - dev(deps_p))

        normal = jax.jacfwd(self.equivalent_stress)

        def r_p(dx):
            deps_p = dx[:-1]
            dp = dx[-1]
            sig_eq = self.equivalent_stress(stress(deps_p))
            r_elastic = lambda dp: dp
            r_plastic = lambda dp: sig_eq - self.yield_stress(p_old + dp)
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)

        def r_eps_p(dx):
            deps_p = dx[:-1]
            dp = dx[-1]

            sig = stress(deps_p)
            n = normal(sig)
            r_elastic = lambda deps_p, dp: deps_p
            r_plastic = lambda deps_p, dp: deps_p - n * dp
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, deps_p, dp)

        self.newton_solver.set_residual((r_eps_p, r_p))

        x0 = jnp.zeros((7,))
        x, data = self.newton_solver.solve(x0)
        depsp = x[:-1]
        dp = x[-1]

        sig = stress(depsp)

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        return sig, state
