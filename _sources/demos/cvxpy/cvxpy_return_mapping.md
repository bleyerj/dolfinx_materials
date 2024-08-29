---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Computing yield surfaces

In this demo, we show how to implement various yield functions of elastoplastic materials using `cvxpy`.

## Problem position

We first set the problem by defining an elastic material and a function which takes a `CvxPyMaterial` as an input, defines radial load paths in the $(\varepsilon_{xx},\varepsilon_{yy})$ space and integrate the material behavior for each load path. The results are then plotted in the $(\sigma_{xx},\sigma_{yy})$ stress space.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dolfinx_materials.jax_materials import LinearElasticIsotropic
from cvxpy_materials import CvxPyMaterial
import cvxpy as cp


E, nu = 70e3, 0.
elastic_model = LinearElasticIsotropic(E, nu)

def plot_stress_paths(material, ax):
    eps = 1e-3
    Nbatch = 21
    theta = np.linspace(0, 2 * np.pi, Nbatch)
    Eps = np.vstack([np.array([eps * np.cos(t), eps * np.sin(t), 0]) for t in theta])
    material.set_data_manager(Eps.shape[0])
    state = material.get_initial_state_dict()

    N = 20
    t_list = np.linspace(0, 1, N)
    Stress = np.zeros((N, Nbatch, 3))

    for i, t in enumerate(t_list[1:]):
        sig, isv, Ct = material.integrate(t * Eps)

        Stress[i + 1, :, :] = sig

        material.data_manager.update()
        
    cmap = plt.get_cmap("inferno")
    for j in range(Nbatch):
        points = Stress[:, [j], :2]
        segments = np.concatenate([points[:-1, :], points[1:, :]], axis=1)

        lc = LineCollection(segments, cmap=cmap, linewidths=1 + t_list * 5)
        lc.set_array(np.linspace(0, N - 1, N))
        ax.add_collection(lc)
    return Stress
```

## von Mises material

We start with the plane stress von Mises yield function which is defined as:

$$
\newcommand{\bsig}{\boldsymbol{\sigma}}
f(\bsig)=\sqrt{\sigma_{xx}^2+\sigma_{yy}^2-\sigma_{xx}\sigma_{yy}+\sigma_{xy}^2} - \sigma_0 \leq 0
$$

For the `cvxpy` implementation, the yield function is first reexpressed as:

$$
\begin{align}
\sqrt{\{\sigma\}^\text{T}\mathbf{Q}\{\sigma\}} &\leq \sigma_0 \\
\text{where } \{\sigma\} &= \begin{Bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \sigma_{xy} \end{Bmatrix} \\
\mathbf{Q} &= \begin{bmatrix} 1 & -1/2 & 0 \\ -1/2 & 1 & 0\\ 0 & 0 & 1\end{bmatrix}
\end{align}
$$

To follow Disciplined Convex Programming rules of `cvxpy`, we finally write $\{\sigma\}^\text{T}\mathbf{Q}\{\sigma\} \leq \sigma_0**2$ which reads:

```{code-cell} ipython3
class PlaneStressvonMises(CvxPyMaterial):
    def yield_constraints(self, sig):
        Q = np.array([[1, -1 / 2, 0], [-1 / 2, 1, 0], [0, 0, 1]])
        sig_eq2 = cp.quad_form(sig, Q)
        return [sig_eq2 <= self.sig0**2]
```

Plotting the result, we obtain:

```{code-cell} ipython3
fig, ax = plt.subplots()
sig0 = 30
material = PlaneStressvonMises(elastic_model, sig0=sig0)
plot_stress_paths(material, ax)
sig = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 100)])
sig_eq = np.sqrt(sig[:, 0] ** 2 + sig[:, 1] ** 2 - sig[:, 0] * sig[:, 1])
yield_surface = sig * sig0 / np.repeat(sig_eq[:, np.newaxis], 2, axis=1)
ax.plot(yield_surface[:, 0], yield_surface[:, 1], "-k", linewidth=0.5)
plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.xlim(-1.2 * sig0, 1.2 * sig0)
plt.ylim(-1.2 * sig0, 1.2 * sig0)
ax.set_aspect("equal")
plt.show()
```

## Rankine material

We consider here a Rankine material characterized by uniaxial tensile and compressive strengths $f_t$ and $f_c$ respectively. The yield condition is expressed as:

$$
f(\bsig) \leq 0 \quad \Leftrightarrow -f_c \leq \sigma_I,\sigma_{II} \leq f_t
$$
where $\sigma_I$ and $\sigma_{II}$ denote the principal values. Equivalently, this can be expressed using the minimum and maximum eigenvalues $\sigma_\text{max} \leq f_t$ and $\sigma_\text{min} \geq -f_c$. The maximum and minimum principal value are obtained with `cvxpy.lambda_max` and `cvxpy.lambda_min` respectively, resulting in the following implementation:

```{code-cell} ipython3
class Rankine(CvxPyMaterial):
    def yield_constraints(self, sig):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        return [
            cp.lambda_max(Sig) <= self.ft,
            cp.lambda_min(Sig) >= -self.fc,
        ]
```

In the plane stress space $(\sigma_{xx},\sigma_{yy},\sigma_{xy}=0)$, the Rankine criterion is a square delimited by $-f_c$ in compression and $+f_t$ in tension.

```{code-cell} ipython3
fig, ax = plt.subplots()
fc, ft = 30.0, 10.0
yield_surface = np.array([[-fc, -fc], [-fc, ft], [ft, ft], [ft, -fc], [-fc, -fc]])
ax.plot(yield_surface[:, 0], yield_surface[:, 1], "-k", linewidth=0.5)
material = Rankine(elastic_model, fc=fc, ft=ft)
plot_stress_paths(material, ax)
plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.xlim(-1.2 * fc, 1.2 * ft)
plt.ylim(-1.2 * fc, 1.2 * ft)
ax.set_aspect("equal")
plt.show()
```

## Hosford material

```{seealso}
For more details on the Hosford yield surface, see also [](/demos/elastoplasticity/Hosford_yield_surface.md).
```

The Hosford yield surface in plane stress conditions is defined by (see also {cite:p}`bleyer2021automated`):

$$
f(\bsig) = \left(\dfrac{1}{2}\left(|\sigma_I|^a+|\sigma_{II}|^a+|\sigma_{I}-\sigma_{II}|^a\right)\right)^{1/a} - \sigma_0 \leq 0
$$

We again use `cp.lambda_max` and `cp.lambda_min` as in the Rankine model. We further introduce additional auxiliary optimization variables $\boldsymbol{z}=(z_0,z_1,z_2)$ such that:

$$
\begin{align}
\text{tr}(\bsig)=\sigma_I+\sigma_{II} &= z_0-z_1\\
\sigma_\text{max}-\sigma_{\text{min}} = |\sigma_I-\sigma_{II}| & \leq z_2\\
z_0+z_1&=z_2
\end{align}
$$

Then we easily see that this implies:

$$\begin{align}
\dfrac{1}{2}\left(\sigma_I+\sigma_{II}+|\sigma_I-\sigma_{II}|\right)=\sigma_\text{max} &\leq z_0\\
\dfrac{1}{2}\left(|\sigma_{I}-\sigma_{II}|-\sigma_I-\sigma_{II}\right)=-\sigma_{\min} & \leq z_1
\end{align}
$$

As a result, the yield condition is equivalent to:

$$
\left(\dfrac{1}{2}\left(|z_0|^a+|z_1|^a+|z_2|^a\right)\right)^{1/a} \leq \sigma_0
$$
in which the last condition can be expressed as a $p$-norm on the vector $\boldsymbol{z}$ with here $p=a$. The implementation reads:

```{code-cell} ipython3
class PlaneStressHosford(CvxPyMaterial):
    def yield_constraints(self, sig):
        Sig = cp.bmat(
            [
                [sig[0], sig[2] / np.sqrt(2)],
                [sig[2] / np.sqrt(2), sig[1]],
            ]
        )
        z = cp.Variable(3)
        return [
            cp.trace(Sig) == z[0] - z[1],
            cp.lambda_max(Sig) - cp.lambda_min(Sig) <= z[2],
            z[2] == z[0] + z[1],
            cp.norm(z, p=self.a) <= 2 ** (1 / self.a) * self.sig0,
        ]
```

We obtain the final Hosford yield surface.

```{code-cell} ipython3
fig, ax = plt.subplots()
sig0 = 30
a = 10
material = PlaneStressHosford(elastic_model, sig0=sig0, a=a)
plot_stress_paths(material, ax)
sig = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 100)])
sig_eq = ((np.abs(sig[:, 0]) ** a + np.abs(sig[:, 1]) ** a + np.abs(sig[:, 0] - sig[:, 1])**a)/2)**(1/a)
yield_surface = sig * sig0 / np.repeat(sig_eq[:, np.newaxis], 2, axis=1)
ax.plot(yield_surface[:, 0], yield_surface[:, 1], "-k", linewidth=0.5)
plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.xlim(-1.2 * sig0, 1.2 * sig0)
plt.ylim(-1.2 * sig0, 1.2 * sig0)
ax.set_aspect("equal")
plt.show()
```

Interestingly, the `cvxpy` implementation is able to handle very large values of $a$, contrary to a simple Newton implementation as in [](/demos/elastoplasticity/Hosford_yield_surface.md).

## References

```{bibliography}
:filter: docname in docnames
```
