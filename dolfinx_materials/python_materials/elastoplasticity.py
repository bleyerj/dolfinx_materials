import numpy as np
from .elasticity import LinearElasticIsotropic
from dolfinx_materials.material import Material
from dolfinx_materials.material.generic import tangent_AD
from dolfinx_materials.solvers import JAXNewton
from .tensors import K
import jax
import jax.numpy as jnp


class ElastoPlasticIsotropicHardening(Material):

    def __init__(self, elastic_model, yield_stress, method="implicit"):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.method = method

    @property
    def internal_state_variables(self):
        return {"p": 1}

    # @tangent_AD
    def constitutive_update(self, eps, state, dt):
        if self.method == "unrolled":
            return self.constitutive_update_unrolled(eps, state, dt)
        elif self.method == "implicit":
            return self.constitutive_update_implicit(eps, state, dt)
        else:
            raise NotImplementedError

    def constitutive_update_unrolled(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # scalar instead of 1-dim vector
        sig_old = state["Stress"]

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
            return jax.lax.cond(yield_criterion <= 0.0, r_elastic, r_plastic, dp)

        @jax.jit
        def deps_p(dp):
            def deps_p_elastic(dp):
                return jnp.zeros(6)

            def deps_p_plastic(dp):
                return 3 / 2 * s_el / sig_eq_el * dp

            return jax.lax.cond(dp == 0.0, deps_p_elastic, deps_p_plastic, dp)

        # jax.debug.print("Start Newton loop")
        newton = JAXNewton(lambda x: jnp.array([r(x[0])]))
        dp = 0.0
        x = jnp.array([dp])
        x, res = newton.solve(x)
        dp = x[0]

        # isig_eq = jnp.minimum(1 / sig_eq_el, 1e6)
        # n_el = s_el / sig_eq  # normal vector
        # depsp = 3 / 2 * s_el / sig_eq * dp

        # dR_dp = jax.grad(self.yield_stress)(p_old + dp)
        # beta = 3 * mu * dp / sig_eq
        # gamma = 3 * mu / (3 * mu + dR_dp)
        # D = 3 * mu * (gamma - beta) * jnp.outer(n_el, n_el) + 2 * mu * beta * K()
        # jax.debug.print(
        #     "beta={b}, dR_dp={g}, sig_eq={s}",
        #     b=beta,
        #     g=dR_dp,
        #     s=sig_eq,
        # )
        # Ct = jax.jacfwd()

        sig = sig_el - 2 * mu * deps_p(dp)

        state["Strain"] = eps_old + deps
        state["p"] += dp
        state["Stress"] = sig
        return C, state

    # @tangent_AD
    def constitutive_update_implicit(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # scalar instead of 1-dim vector
        sig_old = state["Stress"]

        elastic_model = self.elastic_model
        E, nu = elastic_model.E, elastic_model.nu
        lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
        C = self.elastic_model.C
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = jnp.sqrt(3 / 2.0) * jnp.linalg.norm(s_el)
        yield_criterion = sig_eq_el - sig_Y_old

        @jax.jit
        def deps_p(dp):
            def deps_p_elastic(dp):
                return jnp.zeros(6)

            def deps_p_plastic(dp):
                return 3 / 2 * s_el / sig_eq_el * dp

            return jax.lax.cond(dp == 0.0, deps_p_elastic, deps_p_plastic, dp)

        @jax.jit
        def r(x):
            deps_el = x[:6]
            dp = x[-1]
            sig = sig_old + C @ deps_el
            s = K() @ sig
            sig_eq = jnp.sqrt(3 / 2.0) * jnp.linalg.norm(s)
            depsp = 3 / 2 * s_el / sig_eq_el * dp

            def r_elastic(deps_el, dp):
                return (
                    deps_el - deps,
                    jnp.array([dp]),
                )

            def r_plastic(deps_el, dp):
                return (
                    deps_el + depsp - deps,
                    jnp.array(
                        [sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)]
                    ),
                )

            return jax.lax.cond(
                yield_criterion < 0.0, r_elastic, r_plastic, deps_el, dp
            )

        # jax.debug.print("Start Newton loop")
        newton = JAXNewton(lambda x: jnp.concatenate(r(x)))
        x = jnp.zeros((7,))
        x.at[:6].set(deps)

        # newton = JAXNewton(lambda x: jnp.array(r(x[0])))
        # dp = 0.0
        # x = jnp.array([dp])

        # J = jax.jacfwd(lambda x: jnp.concatenate(r(x)))(x)
        # res = jnp.concatenate(r(x))
        # j_inv_vp, info = jax.scipy.sparse.linalg.gmres(J, -res)
        # jax.debug.print("{j}", j=J)
        # j_inv_vp = jnp.linalg.solve(J, -res)
        # j_inv_vp, info = jax.scipy.sparse.linalg.cg(J, -res)
        # x += j_inv_vp

        x, res = newton.solve(x)
        deps_el = x[:6]
        dp = x[-1]

        # jax.debug.print("deps_el={deps_el}, dp={dp}", deps_el=deps_el, dp=dp)

        # isig_eq = jnp.minimum(1 / sig_eq_el, 1e6)
        # n_el = s_el / sig_eq  # normal vector
        # depsp = 3 / 2 * s_el / sig_eq * dp

        # dR_dp = jax.grad(self.yield_stress)(p_old + dp)
        # beta = 3 * mu * dp / sig_eq
        # gamma = 3 * mu / (3 * mu + dR_dp)
        # D = 3 * mu * (gamma - beta) * jnp.outer(n_el, n_el) + 2 * mu * beta * K()
        # jax.debug.print(
        #     "beta={b}, dR_dp={g}, sig_eq={s}",
        #     b=beta,
        #     g=dR_dp,
        #     s=sig_eq,
        # )
        # Ct = jax.jacfwd()
        # indices = jnp.arange(0, 6)
        # J11 = newton.jacobian(x)[jnp.ix_(1 + indices, 1 + indices)]
        # iJ11 = jnp.linalg.inv(newton.jacobian(x))[jnp.ix_(indices, indices)]
        # Ct = J11
        # sig = sig_old + C @ deps_el
        sig = sig_el - 2 * mu * deps_p(dp)

        state["Strain"] = eps_old + deps
        state["p"] += dp
        state["Stress"] = sig
        return C, state

    # def constitutive_update(self, eps, state):
    #     return jax.jacfwd(self.local_constitutive_update, argnums=0, has_aux=True)(
    #         eps, state
    #     )
