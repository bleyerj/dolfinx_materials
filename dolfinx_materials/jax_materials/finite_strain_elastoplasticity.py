import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton
from .tensors import dev, tr, det, to_mat, to_vect


class FeFpJ2Plasticity(JAXMaterial):
    """Material model based on https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6843"""

    def __init__(self, elastic_model, yield_stress, theta=1.0):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_stress = lambda x: jnp.linalg.norm(x)
        self.newton_solver = JAXNewton(rtol=1e-8, atol=1e-8, niter_max=100)
        self.theta = theta

    @property
    def fluxes(self):
        return {"PK1": 9}

    @property
    def gradients(self):
        return {"F": 9}

    @property
    def internal_state_variables(self):
        return {"p": 1, "be_bar": 6}

    # FIXME: need initializer for state variables

    @tangent_AD
    def constitutive_update(self, Fv, state, dt):
        F = to_mat(Fv)
        F_old = to_mat(state["F"])
        p_old = state["p"][0]
        be_bar_old = to_mat(state["be_bar"])

        I = jnp.eye(3)

        # relative strain and elastic predictor
        f = F @ jnp.linalg.inv(F_old)
        f_bar = jnp.linalg.det(f) ** (-1 / 3) * f
        be_bar_trial = f_bar.T @ be_bar_old @ f_bar

        s_trial = self.elastic_model.mu * dev(be_bar_trial)

        # elastic predictor yield criterion
        clipped_equiv_stress = lambda s: jnp.clip(self.equivalent_stress(s), a_min=1e-8)
        sig_eq_trial = clipped_equiv_stress(s_trial)
        yield_criterion = sig_eq_trial - jnp.sqrt(2 / 3) * self.yield_stress(p_old)

        # yield surface normal
        normal = jax.jacfwd(clipped_equiv_stress)

        # plastic yield condition residual
        def r_p(dx):
            be_bar = to_mat(dx[:-1])
            dp = dx[-1]
            s = self.elastic_model.mu * dev(be_bar)
            r_elastic = lambda dp: dp
            r_plastic = (
                lambda dp: (
                    clipped_equiv_stress(s)
                    - jnp.sqrt(2 / 3) * self.yield_stress(p_old + dp)
                )
                / self.elastic_model.E
            )
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)

        # flow rule and plastic incompressibility residual
        def r_be(dx):
            be_bar = to_mat(dx[:-1])
            dp = dx[-1]
            s = self.elastic_model.mu * dev(be_bar)
            r_elastic = lambda be_bar, dp: to_vect(be_bar - be_bar_trial, symmetry=True)
            r_plastic = lambda be_bar, dp: to_vect(
                dev(be_bar - be_bar_trial)
                + 2 * jnp.sqrt(3 / 2) * dp * tr(be_bar) / 3 * normal(s)
                + I * (det(be_bar) - 1),
                symmetry=True,
            )
            return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, be_bar, dp)

        self.newton_solver.set_residual((r_be, r_p))

        # Implicit system solution
        x0 = jnp.zeros((7,))
        x0 = x0.at[:-1].set(to_vect(be_bar_trial, True))
        x, data = self.newton_solver.solve(x0)

        be_bar = to_mat(x[:-1])
        dp = x[-1]

        # Kirchhoff (tau) and PK1 (P) stress
        s = self.elastic_model.mu * dev(be_bar)
        J = det(F)
        tau = s + self.elastic_model.kappa / 2 * (J**2 - 1) * I
        P = to_vect(tau @ jnp.linalg.inv(F).T, symmetry=False)

        state["be_bar"] = to_vect(be_bar, symmetry=True)
        state["p"] += dp
        state["PK1"] = P

        return P, state
