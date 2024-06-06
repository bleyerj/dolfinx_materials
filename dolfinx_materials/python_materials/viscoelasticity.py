import numpy as np
from dolfinx_materials.material import Material
from dolfinx_materials.material.generic import tangent_AD
from .elasticity import LinearElasticIsotropic
import jax.numpy as jnp


class LinearViscoElasticity(Material):

    def __init__(self, branch0, branch1, eta, nud):
        super().__init__()
        self.branch0 = branch0  # should be a LinearElastic material
        self.branch1 = branch1  # should be a LinearElastic material
        self.eta = eta
        self.nud = nud  # dissipative Poisson ratio
        self.Cd = LinearElasticIsotropic(self.eta, self.nud).C

    @property
    def internal_state_variables(self):
        return {"epsv": 6}

    def constitutive_update(self, eps, state, dt):
        return self.constitutive_update_inner(eps, state, dt)

    @tangent_AD
    def constitutive_update_direct(self, eps, state, dt):
        epsv_old = state["epsv"]
        Id = jnp.eye(6)
        iTau = self.branch1.C @ jnp.linalg.inv(self.Cd)
        A = jnp.linalg.inv(Id + dt * iTau)
        epsv_new = A @ (epsv_old + dt * iTau @ eps)
        sig = self.branch0.C @ eps + self.branch1.C @ (eps - epsv_new)
        state["epsv"] = epsv_new
        state["Strain"] = eps
        state["Stress"] = sig
        return sig, state

    # @tangent_AD
    def constitutive_update_inner(self, eps, state, dt):
        epsv_old = state["epsv"]
        Id = jnp.eye(6)
        iTau = self.branch1.C @ jnp.linalg.inv(self.Cd)

        def r(epsv):
            return epsv - epsv_old + dt * iTau @ (epsv - eps)

        import jax

        from dolfinx_materials.solvers import JAXNewton

        newton = JAXNewton(r)
        epsv_new, res = newton.solve(epsv_old)
        indices = jnp.arange(0, 6)
        J11 = newton.jacobian(epsv_new)[jnp.ix_(1 + indices, 1 + indices)]
        iJ11 = jnp.linalg.inv(newton.jacobian(epsv_new))[jnp.ix_(indices, indices)]
        Ct = self.branch0.C + self.branch1.C - self.branch1.C @ J11 @ (dt * iTau)
        # sig = sig_old + C @ deps_el

        # epsv = epsv_old
        # J = jax.jacfwd(r)(epsv)
        # res = r(epsv)
        # res = epsv - epsv_old + dt * iTau @ (epsv - eps)
        # j_inv_vp, info = jax.scipy.sparse.linalg.gmres(J, -res)
        # j_inv_vp = jnp.linalg.solve(J, -res)
        # epsv_new = +j_inv_vp

        sig = self.branch0.C @ eps + self.branch1.C @ (eps - epsv_new)
        state["epsv"] = epsv_new
        state["Strain"] = eps
        state["Stress"] = sig
        return Ct, state
