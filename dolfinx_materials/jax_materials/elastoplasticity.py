import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD
from dolfinx_materials.jax_newton_solver import JAXNewton
from .tensors import dev, to_mat


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


equivalent_norms = {
    "von_Mises": lambda sig: jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig)),
    "Hosford": Hosford_stress,
}


class vonMisesIsotropicHardening(JAXMaterial):
    def __init__(self, elastic_model, yield_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_norm = equivalent_norms["von_Mises"]

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
        sig_eq_el = jnp.clip(self.equivalent_norm(sig_el), a_min=1e-8)
        n_el = dev(sig_el) / sig_eq_el
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
                return 3 / 2 * n_el * dp  # n=n_el simplification due to radial return

            return jax.lax.cond(
                yield_criterion < 0.0,
                deps_p_elastic,
                deps_p_plastic,
                dp,
                yield_criterion,
            )

        newton = JAXNewton(r)
        dp, res = newton.solve(0.0)

        sig = sig_el - 2 * mu * deps_p(dp, yield_criterion)

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        return sig, state


class GeneralIsotropicHardening(JAXMaterial):

    def __init__(self, elastic_model, yield_stress, norm_type="vonMises"):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_norm = equivalent_norms[norm_type]

    @property
    def internal_state_variables(self):
        return {
            "p": 1,
            "eps_p": 6,
        }

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]

        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        sig_old = state["Stress"]

        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = jnp.clip(self.equivalent_norm(sig_el), a_min=1e-8)

        yield_criterion = sig_eq_el - sig_Y_old

        def stress(deps_p):
            return sig_old + C @ (deps - dev(deps_p))

        # normal = jax.jacfwd(self.equivalent_norm)
        def normal(sig):
            sig_eq = jnp.clip(self.equivalent_norm(sig), a_min=1e-8)
            return 3 / 2 * dev(sig) / sig_eq

        def r_p(dx):
            deps_p = dx[:-1]
            dp = dx[-1]
            sig_eq = self.equivalent_norm(stress(deps_p))
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

        newton = JAXNewton((r_eps_p, r_p))

        x0 = jnp.zeros((7,))
        x, res = newton.solve(x0)

        depsp = x[:-1]
        dp = x[-1]

        sig = stress(depsp)

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        # state["eps_p"] += deps_p
        return sig, state
