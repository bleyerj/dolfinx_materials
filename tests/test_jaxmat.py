import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, io

from dolfinx_materials.quadrature_map import QuadratureMap

from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.solvers import NonlinearMaterialProblem


from dolfinx.common import list_timings, TimingType, Timer
import jax

jax.config.update("jax_platform_name", "cpu")

import jaxmat.materials as jm
import jax.numpy as jnp
import equinox as eqx
from jax_material import JAXMaterial

elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)


class VoceHardening(eqx.Module):
    sig0: float
    sigu: float
    b: float

    def __call__(self, p):
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-self.b * p))


hardening = VoceHardening(sig0=350.0, sigu=500.0, b=1e3)

behavior = jm.vonMisesIsotropicHardening(elasticity=elasticity, yield_stress=hardening)


material = JAXMaterial(behavior)
material_prop = material.material_properties

order = 2
N = 50

domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("P", order, (2,)))

deg_quad = 2 * (order - 1)


def bottom(x):
    return np.isclose(x[1], 0)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1.0)


V_ux, _ = V.sub(0).collapse()
left_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), left)
right_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), right)
V_uy, _ = V.sub(1).collapse()
bottom_dofs_uy = fem.locate_dofs_geometrical((V.sub(1), V_uy), bottom)

uD_x = fem.Function(V_ux)
uD_y = fem.Function(V_uy)
uD_x_r = fem.Function(V_ux)
bcs = [
    fem.dirichletbc(uD_x, left_dofs_ux, V.sub(0)),
    fem.dirichletbc(uD_y, bottom_dofs_uy, V.sub(1)),
    fem.dirichletbc(uD_x_r, right_dofs_ux, V.sub(0)),
]

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V)
# u.x.array[:] = np.random.rand(len(u.x.array))


def strain(u):
    return ufl.as_vector(
        [
            u[0].dx(0),
            u[1].dx(1),
            0.0,
            1 / np.sqrt(2) * (u[1].dx(0) + u[0].dx(1)),
            0.0,
            0.0,
        ]
    )


qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient(material.gradient_names[0], strain(u))

sig = qmap.fluxes[material.flux_names[0]]
Res = ufl.dot(sig, strain(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)
newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6
newton.max_it = 20

t = Timer("First call to update")
qmap.update()
t.stop()

Exx = np.linspace(0, 1e-2, 10)
# if save_fields:
#     file_results = io.XDMFFile(
#         domain.comm,
#         f"{material.name}_results.xdmf",
#         "w",
#     )
#     file_results.write_mesh(domain)
Stress = np.zeros((len(Exx), 6))
for i, exx in enumerate(Exx[1:]):
    uD_x_r.x.array[:] = exx

    converged, nits = problem.solve(newton, False)

    assert converged
    print("Converged in ", nits)
    Stress[i + 1, :] = sig.x.array[:6]

    # if save_fields:
    #     for field_name in save_fields:
    #         field = qmap.project_on(field_name, ("DG", 0))
    #         file_results.write_function(field, i)
list_timings(domain.comm, [TimingType.wall, TimingType.user])
# import matplotlib.pyplot as plt

# plt.plot(Exx, Stress[:, 0])
# plt.show()
