import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dolfinx_materials.jax_materials import LinearElasticIsotropic
from cvxpy_materials import Rankine, PlaneStressvonMises, PlaneStressHosford


E, nu = 70e3, 0.2
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

    # Create a colormap
    cmap = plt.get_cmap("inferno")
    for j in range(Nbatch):
        points = Stress[:, [j], :2]
        segments = np.concatenate([points[:-1, :], points[1:, :]], axis=1)

        lc = LineCollection(segments, cmap=cmap, linewidths=1 + t_list * 5)
        lc.set_array(np.linspace(0, N - 1, N))
        ax.add_collection(lc)
    return Stress


fig, ax = plt.subplots()
sig0 = 30
material = PlaneStressHosford(elastic_model, sig0=sig0, a=10)
plot_stress_paths(material, ax)
plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.xlim(-1.2 * sig0, 1.2 * sig0)
plt.ylim(-1.2 * sig0, 1.2 * sig0)
plt.gca().set_aspect("equal")
plt.show()


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
plt.gca().set_aspect("equal")
plt.show()


fig, ax = plt.subplots()
sig0 = 30
material = PlaneStressvonMises(elastic_model, sig0=sig0)
plot_stress_paths(material, ax)
plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.xlim(-1.2 * sig0, 1.2 * sig0)
plt.ylim(-1.2 * sig0, 1.2 * sig0)
plt.gca().set_aspect("equal")
plt.show()
