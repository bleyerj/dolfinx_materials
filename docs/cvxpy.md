# Convex optimization-based constitutive update

## Elastoplastic return mapping via convex optimization

For a specific class of materials, constitutive update can be also formulated as solving a convex optimization problem. In the case of associated plasticity, this amounts to projecting the elastic trial state onto the yield surface. Ignoring hardening for the sake of simplicity, such a projection can be formulated as follows:

```{math}
:label: stress-projection

\newcommand{\bsig}{\boldsymbol{\sigma}}
\newcommand{\beps}{\boldsymbol{\varepsilon}}
\begin{array}{rl}
\displaystyle{\min_{\bsig}} & \dfrac{1}{2}(\bsig-\bsig_\text{el}):\mathbb{C}^{-1}:(\bsig-\bsig_\text{el})\\
\text{s.t.} & f(\bsig) \leq 0
\end{array}
```

where $\bsig_\text{el} = \bsig_\text{old} + \mathbb{C}:\Delta\beps$ is the elastic predictor computed from the previous stress state $\bsig_\text{old}$ and the current strain increment $\Delta\beps$.

Writing the optimality conditions of {eq}`stress-projection` results in the system of nonlinear equations arising in the elastoplastic return mapping procedure. However, when considering non-smooth yield surfaces or yield conditions formed by the intersection of many yield surfaces, such local Newton procedures may encounter difficulties of convergence, due to the lack of differentiability of the yield surface.

In such cases, it may become interesting to consider using dedicated tools to solve the optimization problem {eq}`stress-projection` directly. At first sight, this problem is a quadratic optimization problem with nonlinear constraints. However, the yield function $f(\bsig)$ can very often be reformulated using standard cones such as the positive orthant, the second-order Lorentz cone, the cone of positive-definite matrices, etc. In this case, we can leverage more efficient convex optimization solvers such as primal-dual interior point solvers.

## CVXPY

[`cvxpy`](https://www.cvxpy.org) is an open source Python-embedded modeling language for convex optimization problems, enabling to express such problems in a natural way using predefined atomic functions. It offers a wide range of open-source and commercial solvers. Some of them are also capable of computing sensitivities of the optimization problem to derive the underlying tangent operator. This is however not explored here at the moment.

```{seealso}
For more details, see also Andrey's Latyshev work at https://github.com/a-latyshev/convex-plasticity.
```

## Implementation

We describe below the implementation of a base class called `CvxPyMaterial`. For simplicity, such materials are limited to plane stress elastoplasticity without hardening. We first initialize the class which inherits from the generic `Material` and enforce the plane dimension of stresses and strains. 

```python
from dolfinx_materials.material import Material
import cvxpy as cp
import numpy as np


class CvxPyMaterial(Material):
    def __init__(self, elastic_model, **kwargs):
        super().__init__()
        self.elastic_model = elastic_model

        # Handle any additional keyword arguments passed
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.C = np.array(self.elastic_model.compute_C_plane_stress())
        self.set_cvxpy_model()

    @property
    def gradients(self):
        return {"Strain": 3}

    @property
    def fluxes(self):
        return {"Stress": 3}
```

The implementation of the optimization problem with `cvxpy` is done in the `set_cvxpy_model` method. The optimization variable is a stress state of dimension 3. The elastic stress is defined as a `cvxpy.Parameter` which are symbolic representations of constants. Using parameters allows to modify the values of constants without reconstructing the entire problem. However, this places some restriction on the way the problem is defined. These aspects are described in the [Disciplined Parametrized Programming](https://www.cvxpy.org/tutorial/dpp/index.html) section of `cvxpy` documentation. Here, we define the objective function `obj` corresponding to $\dfrac{1}{2}(\bsig-\bsig_\text{el}):\mathbb{C}^{-1}:(\bsig-\bsig_\text{el})$. The constraints are provided by the `yield_constraints` method which returns an empty list at the moment. Concrete implementation of subclasses of `CvxPyMaterial` should implementation the constraints corresponding to $f(\bsig) \leq 0$.

```python
    def set_cvxpy_model(self):
        self.sig = cp.Variable((3,))
        self.sig_el = cp.Parameter((3,))
        obj = 0.5 * cp.quad_form(self.sig - self.sig_el, np.linalg.inv(self.C))
        self.prob = cp.Problem(cp.Minimize(obj), self.yield_constraints(self.sig))

    def yield_constraints(self, Sig):
        return []
```

Finally, the constitutive update rule is defined by computing the elastic predictor, affecting its value to the `cvxpy.Parameter` object and solving the problem. At the moment, this function returns the elastic operator only.

```python
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        sig_old = state["Stress"]

        sig_pred = sig_old + self.C @ deps

        self.sig_el.value = sig_pred
        self.prob.solve()

        state["Strain"] = eps
        state["Stress"] = self.sig.value
        return self.C, state
 ```