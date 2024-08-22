import numpy as np
from dolfinx_materials.jax_materials.elasticity import (
    PlaneStressLinearElasticIsotropic,
)
from rankine import Rankine
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, io
import matplotlib.pyplot as plt

from dolfinx_materials.quadrature_map import QuadratureMap

from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx.cpp.nls.petsc import NewtonSolver

E, nu = 70e3, 0.0
fc, ft = 30.0, 30.0
elastic_model = PlaneStressLinearElasticIsotropic()
elastic_model.C = elastic_model.compute_C(E, nu)

material = Rankine(elastic_model, fc, ft)

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("P", 1, (2,)))

deg_quad = 0

Eps = fem.Constant(domain, np.zeros((3,)))
x = ufl.SpatialCoordinate(domain)
Eps_t = ufl.as_matrix([[Eps[0], Eps[2]], [Eps[2], Eps[1]]])

u_expr = fem.Expression(Eps_t * x, V.element.interpolation_points())
uD = fem.Function(V)
uD.interpolate(u_expr)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bcs = [fem.dirichletbc(uD, boundary_dofs)]

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V)


def strain(u):
    return ufl.as_vector(
        [
            u[0].dx(0),
            u[1].dx(1),
            1 / np.sqrt(2) * (u[1].dx(0) + u[0].dx(1)),
        ]
    )


qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("Strain", strain(u))
sig = qmap.fluxes["Stress"]
Res = ufl.dot(sig, strain(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)
newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6

theta_list = np.linspace(0, 2 * np.pi, 15)[:-1]


t_list = np.linspace(0, 1e-3, 25)
Stress = np.zeros((len(t_list), len(theta_list), 3))


fig = plt.figure()
yield_surface = np.array([[-fc, -fc], [-fc, ft], [ft, ft], [ft, -fc], [-fc, -fc]])
plt.plot(yield_surface[:, 0], yield_surface[:, 1], "-k", linewidth=0.5)
c = np.linspace(0, 1, len(t_list))
colors = plt.cm.inferno(c)

for j, t in enumerate(theta_list):
    Eps_dir = np.array([np.cos(t), np.sin(t), 0.0])
    sig.x.set(0)
    u.x.set(0)
    qmap.reinitialize_state()

    for i, t in enumerate(t_list[1:]):
        Eps.value = t * Eps_dir
        uD.interpolate(u_expr)

        converged, it = problem.solve(newton)
        Stress[i + 1, j, :] = sig.vector.array[:3]

        plt.scatter(Stress[:, j, 0], Stress[:, j, 1], marker="o", c=colors[:])

plt.xlabel(r"Stress $\sigma_{xx}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.gca().set_aspect("equal")
plt.show()
