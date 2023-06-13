import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, io, fem
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx.common import list_timings, TimingType
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.python_materials import (
    ElastoPlasticIsotropicHardening,
    LinearElasticIsotropic,
)
from generate_mesh import generate_perforated_plate

hypothesis = "plane_strain"
# hypothesis = "plane_stress"

E = 70e3
H = E / 1e3
sig0 = 500.0
elastic_model = LinearElasticIsotropic(E=70e3, nu=0.3)


def yield_stress(p):
    return sig0 + H * p


material = ElastoPlasticIsotropicHardening(
    elastic_model=elastic_model, yield_stress=yield_stress
)
N = 15
Eyy = np.linspace(0, 10e-3, N + 1)

W = 1.0
H = 2.0
R = 0.2
mesh_size = 0.05

generate_perforated_plate(W, H, R, [0.02, 0.2])

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as infile:
    domain = infile.read_mesh(mesh.GhostMode.none)


def bottom(x):
    return np.isclose(x[1], 0)


def top(x):
    return np.isclose(x[1], H)


top_facets = mesh.locate_entities(domain, 1, top)

facet_tag = mesh.meshtags(
    domain, 1, top_facets, np.full_like(top_facets, 1, dtype=np.int32)
)
ds = ufl.Measure("ds", subdomain_data=facet_tag)

order = 2
deg_quad = 2 * (order - 1)
if hypothesis == "plane_strain":
    V = fem.VectorFunctionSpace(domain, ("CG", order))
    top_dofs = fem.locate_dofs_geometrical(V, top)
    bottom_dofs = fem.locate_dofs_geometrical(V, bottom)

    uD_b = fem.Function(V)
    uD_t = fem.Function(V)
    bcs = [fem.dirichletbc(uD_t, top_dofs), fem.dirichletbc(uD_b, bottom_dofs)]

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

elif hypothesis == "plane_stress":
    Ue = ufl.VectorElement("CG", domain.ufl_cell(), order)
    Ee = ufl.FiniteElement("DG", domain.ufl_cell(), order - 1)
    V = fem.FunctionSpace(domain, ufl.MixedElement([Ue, Ee]))

    V_u, mapping = V.sub(0).collapse()
    top_dofs = fem.locate_dofs_geometrical((V.sub(0), V_u), top)
    bottom_dofs = fem.locate_dofs_geometrical((V.sub(0), V_u), bottom)

    uD_b = fem.Function(V_u)
    uD_t = fem.Function(V_u)
    bcs = [
        fem.dirichletbc(uD_t, top_dofs, V.sub(0)),
        fem.dirichletbc(uD_b, bottom_dofs, V.sub(0)),
    ]

    def strain(v):
        u, ezz = ufl.split(v)
        return ufl.as_vector(
            [
                u[0].dx(0),
                u[1].dx(1),
                ezz,
                1 / np.sqrt(2) * (u[1].dx(0) + u[0].dx(1)),
                0.0,
                0.0,
            ]
        )

else:
    raise ValueError(
        "Wrong hypothesis type. Only plane_stress/plane_strain is supported."
    )

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V)


qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient(material.gradient_names[0], strain(u))

sig = qmap.fluxes["Stress"]
Res = ufl.dot(sig, strain(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6
newton.convergence_criterion = "incremental"
newton.report = True


file_results = io.XDMFFile(
    domain.comm,
    f"{hypothesis}_plasticity_results.xdmf",
    "w",
)
file_results.write_mesh(domain)
Syy = np.zeros_like(Eyy)
for i, eyy in enumerate(Eyy[1:]):
    uD_t.vector.array[1::2] = eyy * H

    converged, it = problem.solve(newton)

    p = qmap.project_on("p", ("DG", 0))
    stress = qmap.project_on("Stress", ("DG", 0))
    file_results.write_function(p, i)
    file_results.write_function(stress, i)

    Syy[i + 1] = fem.assemble_scalar(fem.form(stress[1] * ds(1))) / W
    print(Syy)


list_timings(domain.comm, [TimingType.wall, TimingType.user])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(Eyy, Syy, "-o")
plt.xlabel(r"Strain $\varepsilon_{yy}$")
plt.ylabel(r"Stress $\sigma_{yy}$")
plt.savefig(f"{material.name}_stress_strain.pdf")
res = np.zeros((len(Eyy), 2))
res[:, 0] = Eyy
res[:, 1] = Syy
np.savetxt(f"{hypothesis}_plasticity_results.csv", res, delimiter=",")
