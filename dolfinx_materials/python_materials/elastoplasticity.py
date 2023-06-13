import numpy as np
from .elasticity import LinearElasticIsotropic
from dolfinx_materials.material import Material
from .tensors import Identity, K, vectorized_outer, expand_size
from scipy.optimize import root, fsolve

# import jax.numpy as jnp
# from jax import jit

# from functools import partial
# from jax import lax
# from jaxopt import Bisection


class ElastoPlasticIsotropicHardening(Material):
    @property
    def internal_state_variables(self):
        return {"p": 1}

    def constitutive_update_vectorized_backup(self, eps, state):  # vectorized version
        batch_size = eps.shape[0]

        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"]
        sig_old = state["Stress"]

        elastic_model = self.elastic_model
        E, nu = elastic_model.E, elastic_model.nu
        lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
        C = self.elastic_model.compute_C(E, nu)

        print(eps.shape)
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = np.zeros_like(p_old)
        sig_eq_el[0, :] = np.sqrt(3 / 2.0) * np.linalg.norm(s_el, axis=0)
        yield_criterion = sig_eq_el - sig_Y_old
        Ct = expand_size(C, batch_size)
        dp = np.zeros_like(p_old)
        plastic_state = (yield_criterion >= 0).flatten()
        print(sig_Y_old.shape)
        print(sig_eq_el.shape)
        print((3 * mu * dp).shape)
        print(self.yield_stress(p_old + dp).shape)

        def residual(dp):
            return (
                sig_eq_el.flatten()[plastic_state]
                - 3 * mu * dp
                - self.yield_stress(p_old.flatten()[plastic_state] + dp)
            ).flatten()

        print("res", residual(dp.flatten()[plastic_state]).shape)
        dp.ravel()[plastic_state] = root(
            lambda dp: residual(dp),
            dp.flatten()[plastic_state],
        ).x
        n_el = s_el / sig_eq_el  # normal vector
        depsp = 3 / 2 * n_el * dp
        sig_Y_new = self.yield_stress(p_old + dp)
        dR_dp = (sig_Y_new - sig_Y_old) / dp
        beta = 1 - sig_Y_new / sig_eq_el
        gamma = 3 * mu / (3 * mu + dR_dp)
        D = 3 * mu * (gamma - beta) * vectorized_outer(
            n_el, n_el
        ) + 2 * mu * beta * expand_size(K(), batch_size)
        Ct[:, :, plastic_state] -= D[:, :, plastic_state]
        # else:
        #     dp = 0
        #     depsp = 0 * s_el
        sig = sig_el - 2 * mu * K() @ depsp

        state["Strain"] = eps_old + deps
        state["p"] = p_old + dp
        state["Stress"] = sig
        return sig, Ct

    def constitutive_update(self, eps, state):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"]
        sig_old = state["Stress"]

        elastic_model = self.elastic_model
        E, nu = elastic_model.E, elastic_model.nu
        lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
        C = self.elastic_model.compute_C(E, nu)
        sig_el = sig_old + C @ deps
        s_el = K() @ sig_el
        sig_Y_old = self.yield_stress(p_old)
        sig_eq_el = np.sqrt(3 / 2.0) * np.linalg.norm(s_el)
        print(sig_eq_el)
        yield_criterion = sig_eq_el - sig_Y_old
        if yield_criterion > 0:
            dp = fsolve(
                lambda dp: sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp), 0.0
            )
            # assert dp > 0, f"Wrong value for dp={dp}"
            n_el = s_el / sig_eq_el  # normal vector
            depsp = 3 / 2 * n_el * dp
            sig_Y_new = self.yield_stress(p_old + dp)
            dR_dp = (sig_Y_new - sig_Y_old) / dp
            beta = 1 - sig_Y_new / sig_eq_el
            gamma = 3 * mu / (3 * mu + dR_dp)
            D = 3 * mu * (gamma - beta) * np.outer(n_el, n_el) + 2 * mu * beta * K()
            Ct = C - D
        else:
            dp = 0
            depsp = 0 * s_el
            Ct = C
        sig = sig_el - 2 * mu * K() @ depsp
        print(p_old + dp)
        state["Strain"] = eps_old + deps
        state["p"] = p_old + dp
        state["Stress"] = sig
        return sig, Ct

    # def constitutive_update(self, eps, state):
    #     elastic_model = self.elastic_model
    #     E, nu = elastic_model.E, elastic_model.nu
    #     lmbda, mu = elastic_model.get_Lame_parameters(E, nu)
    #     C = self.elastic_model.compute_C(E, nu)

    #     sig, eps, p, Ct = jax_constitutive_update(
    #         eps,
    #         state,
    #         C,
    #         mu,
    #         K(),
    #     )

    #     state["Strain"] = eps
    #     state["p"] = p
    #     state["Stress"] = sig
    #     return sig, Ct


# @jit
# def jax_constitutive_update(eps, state, C, mu, K):
#     eps_old = state["Strain"]
#     deps = eps - eps_old
#     p_old = state["p"]
#     sig_old = state["Stress"]

#     def yield_stress(p):
#         sig0 = 500.0
#         sigu = 510.0
#         omega = 100.0
#         return sigu + (sig0 - sigu) * jnp.exp(-p * omega)

#     sig_el = sig_old + C @ deps
#     s_el = K @ sig_el
#     sig_Y_old = yield_stress(p_old)
#     sig_eq_el = jnp.sqrt(3 / 2.0) * jnp.linalg.norm(s_el)
#     yield_criterion = sig_eq_el - sig_Y_old

#     def plastic_flow(x):
#         sig_eq_el, mu, p_old = x
#         dp = Bisection(
#             lambda dp: sig_eq_el - 3 * mu * dp - yield_stress(p_old + dp),
#             0.0,
#             1e3,
#         ).run()
#         n_el = s_el / sig_eq_el  # normal vector
#         depsp = 3 / 2 * n_el * dp
#         sig_Y_new = yield_stress(p_old + dp)
#         dR_dp = (sig_Y_new - sig_Y_old) / dp
#         beta = 1 - sig_Y_new / sig_eq_el
#         gamma = 3 * mu / (3 * mu + dR_dp)
#         D = 3 * mu * (gamma - beta) * jnp.outer(n_el, n_el) + 2 * mu * beta * K()
#         Ct = C - D
#         return dp, depsp, Ct

#     def elastic_flow(x):
#         return 0, 0 * deps, C

#     dp, depsp, Ct = lax.cond(
#         yield_criterion[0] >= 0, plastic_flow, elastic_flow, (sig_eq_el, mu, p_old)
#     )

#     # if yield_criterion >= 0:
#     #     dp = fsolve(lambda dp: sig_eq_el - 3 * mu * dp - yield_stress(p_old + dp), 0.0)
#     #     n_el = s_el / sig_eq_el  # normal vector
#     #     depsp = 3 / 2 * n_el * dp
#     #     sig_Y_new = yield_stress(p_old + dp)
#     #     dR_dp = (sig_Y_new - sig_Y_old) / dp
#     #     beta = 1 - sig_Y_new / sig_eq_el
#     #     gamma = 3 * mu / (3 * mu + dR_dp)
#     #     D = 3 * mu * (gamma - beta) * jnp.outer(n_el, n_el) + 2 * mu * beta * K()
#     #     Ct = C - D
#     # else:
#     #     dp = 0
#     #     depsp = 0 * s_el
#     #     Ct = C
#     sig = sig_el - 2 * mu * K @ depsp
#     return sig, eps_old + deps, p_old + dp, Ct
