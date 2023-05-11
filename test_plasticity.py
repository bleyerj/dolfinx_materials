import numpy as np
import ufl
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.materials.python import (
    LinearElasticIsotropic,
    ElastoPlasticIsotropicHardening,
)
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot, la
from dolfinx.fem.petsc import create_matrix
import matplotlib.pyplot as plt

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, mesh.CellType.quadrilateral)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

deg_quad = 1


def bottom(x):
    return np.isclose(x[1], 0)


def left(x):
    return np.isclose(x[0], 0)


def middle(x):
    return np.isclose(x[0], 0.5)


def right(x):
    return np.isclose(x[0], 1.0)


left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)
middle_dofs = fem.locate_dofs_geometrical(V, middle)

Eps = fem.Constant(domain, ((0.0, 0.0)))
Eps2 = fem.Constant(domain, ((0.0, 0.0)))
bcs = [
    fem.dirichletbc(np.zeros((2,)), left_dofs, V),
    fem.dirichletbc(Eps, right_dofs, V),
    # fem.dirichletbc(Eps2, middle_dofs, V),
]


V_ux, mapping = V.sub(1).collapse()
left_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), left)
right_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), right)
V_uy, mapping = V.sub(1).collapse()
bottom_dofs_uy = fem.locate_dofs_geometrical((V.sub(1), V_uy), bottom)

Eps = fem.Constant(domain, 0.0)
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
# x = ufl.SpatialCoordinate(domain)
# expr = fem.Expression(ufl.as_vector([x[0], 0]), V.element.interpolation_points())
# u.interpolate(expr)


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


elastic_model = LinearElasticIsotropic(70e3, 0.3)
sig0 = 500.0
sigu = 750.0
omega = 100.0


def yield_stress(p):
    return sigu + (sig0 - sigu) * np.exp(-p * omega)
    # return 100.0 + 0e3 * p


mat_elastoplastic = ElastoPlasticIsotropicHardening(elastic_model, yield_stress)


qmap = QuadratureMap(domain, deg_quad, strain(u), mat_elastoplastic)
eps = qmap.add_parameter(dim=6, name="eps")
eps_p = qmap.add_parameter(dim=6, name="eps_p")
p = qmap.add_parameter(name="p")
sig = qmap.add_parameter(dim=6, name="sig")
print(qmap.parameters)


cells = qmap.cells
Res = ufl.dot(qmap.flux, strain(v)) * qmap.dx
Jac = ufl.dot(strain(du), ufl.dot(qmap.jacobian, strain(v))) * qmap.dx

from dolfinx_materials.solvers import SNESProblem, CustomNewton


problem = SNESProblem(qmap, Res, Jac, u, bcs)


b = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
J = create_matrix(problem.a)
# Create Newton solver and solve
snes = PETSc.SNES().create()
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-6, atol=1.0e-6, max_it=1000)
snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-6)

newton = CustomNewton(qmap, Res, Jac, u, bcs)
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


reshist = {}


def monitor(ksp, its, rnorm):
    print(f"Iteration {its}  Residual:", rnorm)
    reshist[its] = rnorm


snes.getKSP().setMonitor(monitor)
snes.getKSP().getPC().setType("lu")

N = 10
Exx = np.concatenate(
    (
        np.linspace(0, 2e-2, N + 1),
        np.linspace(2e-2, 1e-2, N + 1)[1:],
        np.linspace(1e-2, 3e-2, N + 1)[1:],
    )
)
# Exx = np.linspace(0, 1e-2, 20)
Sxx = np.zeros_like(Exx)
for i, exx in enumerate(Exx[1:]):
    uD_x_r.vector.array[:] = exx
    print("Exx=", exx)
    sxx = elastic_model.E * exx
    print("Sxx=", sxx)
    # it = snes.solve(None, u.vector)
    converged, it = newton.solve(solver)
    print("Flux", qmap.flux.vector.array)
    print(f"Finished in {it} iterations.")
    Sxx[i + 1] = qmap.flux.vector.array[0]
    # break

plt.figure()
plt.plot(Exx, Sxx, "-o")
plt.xlabel(r"Strain $\varepsilon_{xx}$")
plt.ylabel(r"Stress $\sigma_{xx}$")
plt.savefig("stress_strain_plot.pdf")
