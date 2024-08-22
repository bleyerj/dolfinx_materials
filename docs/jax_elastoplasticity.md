# JAX implementation of elastoplasticity

To go further in the implementation of complex behaviors using JAX, we now describe the implementation of elastoplastic behaviors.

## von Mises plasticity with isotropic hardening

```python
import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton
from .tensors import dev, to_mat

class vonMisesIsotropicHardening(JAXMaterial):
    def __init__(self, elastic_model, yield_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_norm = lambda sig: jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig))

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
```

We declare a scalar plastic variable representing the cumulated plastic strain

We compute an elastic predictor and set a minimum value to its equivalent norm to avoid division by zero.

```python
sig_el = sig_old + C @ deps
sig_eq_el = jnp.clip(self.equivalent_norm(sig_el), a_min=1e-8)
```

We define the normal vector $n_\text{el}$ corresponding to the elastic predictor.
```python
n_el = dev(sig_el) / sig_eq_el
```

We use `JAXNewton` to set up a local Newton solver written in JAX, see here for more details
```python
newton = JAXNewton(r)
dp, res = newton.solve(0.0)
```

Finally, we compute the final stress based on the computed value of the plastic strain increment and we update the state dictionary

```python
sig = sig_el - 2 * mu * deps_p(dp, yield_criterion)

state["p"] += dp
state["Strain"] += deps
state["Stress"] = sig
```

% TODO: JAXNewton