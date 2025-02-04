# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Finite-strain $\boldsymbol{F}^\text{e}\boldsymbol{F}^\text{p}$ plasticity
#
# $\newcommand{\bF}{\boldsymbol{F}}\newcommand{\bFe}{\boldsymbol{F}^\text{e}}\newcommand{\bFp}{\boldsymbol{F}^\text{p}}$ In this example, we show how to use a JAX implementation of finite-strain plasticity using the $\bFe\bFp$ formalism. The setup of the FEniCSx variational problem is quite similar to the MFront [](/demos/mfront/hyperelasticity/hyperelasticity.md) demo.

# +
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.jax_materials import LinearElasticIsotropic, FeFpJ2Plasticity
import jax.numpy as jnp
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector

comm = MPI.COMM_WORLD
rank = comm.rank

dim = 3
L = 10.0  # rod half-length
W = 2.0  # rod diameter
R = 20.0  # notch radius
d = 0.2  # cross-section reduction
coarse_size = 1.0
fine_size = 0.2

# -

# The mesh consists of a quarter of cylindrical rod with a slightly reduced cross-section at its center to induce necking. The geometry is defined with `gmsh` API and the Open-Cascade kernel.

# +
# # + tags=["hide-input"]
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info

if rank == 0:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, W)
    tool = gmsh.model.occ.addDisk(0, W - d + R, 0, R, R)
    vol_dimTag = (dim - 1, rectangle)
    out = gmsh.model.occ.cut([vol_dimTag], [(dim - 1, tool)])

    out = gmsh.model.occ.revolve(out[0], 0, 0, 0, 1, 0, 0, np.pi / 2)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim, [1], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(dim - 1, [2], 1, name="bottom")
    gmsh.model.addPhysicalGroup(dim - 1, [6], 2, name="side_Y")
    gmsh.model.addPhysicalGroup(dim - 1, [1], 3, name="side_Z")
    gmsh.model.addPhysicalGroup(dim - 1, [5], 4, name="top")

    field_tag = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field_tag, "VIn", fine_size)
    gmsh.model.mesh.field.setNumber(field_tag, "VOut", coarse_size)
    gmsh.model.mesh.field.setNumber(field_tag, "XMin", -W)
    gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
    gmsh.model.mesh.field.setNumber(field_tag, "YMin", 0)
    gmsh.model.mesh.field.setNumber(field_tag, "YMax", W)
    gmsh.model.mesh.field.setNumber(field_tag, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(field_tag, "ZMax", W)
    gmsh.model.mesh.field.setNumber(field_tag, "Thickness", W)

    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    gmsh.model.mesh.generate(dim)

domain, subdomains, facets = io.gmshio.model_to_mesh(gmsh.model, comm, 0)

gmsh.finalize()
# -

# Horizontal traction with free transverse displacement is imposed at the rod extremities, symmetry boundary conditions are enforced on other plane surfaces.

# +
fdim = dim - 1
domain.topology.create_connectivity(fdim, dim)

order = 2
V = fem.functionspace(domain, ("P", order, (dim,)))
deg_quad = 2 * (order - 1)
V_x, _ = V.sub(0).collapse()
V_y, _ = V.sub(1).collapse()
V_z, _ = V.sub(2).collapse()


bottom_x_dofs = fem.locate_dofs_topological((V.sub(0), V_x), fdim, facets.find(1))
top_x_dofs = fem.locate_dofs_topological((V.sub(0), V_x), fdim, facets.find(4))
side_y_dofs = fem.locate_dofs_topological((V.sub(1), V_y), fdim, facets.find(2))
side_z_dofs = fem.locate_dofs_topological((V.sub(2), V_z), fdim, facets.find(3))

u0_x = fem.Function(V_x)
u0_y = fem.Function(V_y)
u0_z = fem.Function(V_z)
uD_x = fem.Function(V_x)
bcs = [
    fem.dirichletbc(u0_y, bottom_x_dofs, V.sub(0)),
    fem.dirichletbc(u0_y, side_y_dofs, V.sub(1)),
    fem.dirichletbc(u0_z, side_z_dofs, V.sub(2)),
    fem.dirichletbc(uD_x, top_x_dofs, V.sub(0)),
]
# -

# As in the MFront hyperelastic demo, the nonlinear residual based on PK1 stress is used. The JAX `FeFpJ2Plasticity` is defined based on an elastic material and a user-defined yield stress function.

# +
Id = ufl.Identity(dim)


def F(u):
    return nonsymmetric_tensor_to_vector(Id + ufl.grad(u))


def dF(u, v):
    return ufl.derivative(F(u), u, v)


du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")

E = 70e3
nu = 0.3
sig0 = 500.0

b = 10.0
sigu = 750.0


def yield_stress(p):
    return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))


elastic_model = LinearElasticIsotropic(E, nu)

material = FeFpJ2Plasticity(elastic_model, yield_stress)

qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("F", F(u))

P = qmap.fluxes["PK1"]
Res = ufl.dot(P, dF(u, v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)
# -

# This material model relies on an elastic left Cauchy-Green internal state variable $\overline{\boldsymbol{b}}^\text{e}$ describing the intermediate elastic configuration. The latter must be initialized with the identity tensor to specify a natural unstressed elastic configuration. We use a first call to the material update function to enforce compilation of the JAX behavior.

qmap.initialize_state()
qmap.update_initial_state("be_bar", fem.Constant(domain, (1.0, 1.0, 1.0, 0, 0, 0)))
qmap.update()

# ## Resolution
#
# As in the MFront demo, we define the custom nonlinear problem, the corresponding Newton solver, the PETSc Krylov solver and its Geometric Algebraic MultiGrid preconditioner.

# +
problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)

newton = NewtonSolver(comm)
newton.rtol = 1e-8
newton.atol = 1e-8
newton.convergence_criterion = "residual"

# Set solver options
ksp = newton.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1e-8
opts[f"{option_prefix}pc_type"] = "gamg"
ksp.setFromOptions()
# -

# We loop over an imposed increasing horizontal strain and solve the nonlinear problem. We output the displacement and equivalent plastic strain variables. Finally, we measure the time spent by each rank on the constitutive update and the linear solve and print the average values on rank 0.

# +
# # + tags=["hide-output"]


from dolfinx.common import timing

file_results = io.VTKFile(
    domain.comm,
    f"results/{material.name}.pvd",
    "w",
)

N = 40
Exx = np.linspace(0, 30e-3, N + 1)
for i, exx in enumerate(Exx[1:]):
    uD_x.x.petsc_vec.set(exx * L)

    converged, it = problem.solve(newton)

    p = qmap.project_on("p", ("DG", 0))
    p.name = "EquivalentPlasticStrain"

    file_results.write_function(u, i + 1)
    file_results.write_function(p, i + 1)

    constitutive_update_time = timing("Constitutive update")[2]
    linear_solve_time = timing("PETSc Krylov solver")[2]

    # Gather all times on rank 0
    all_times = None
    if rank == 0:
        all_times = np.zeros((comm.size, 2))
    comm.Gather(
        np.array([constitutive_update_time, linear_solve_time]), all_times, root=0
    )

    # Compute the average time on rank 0
    if rank == 0:
        average_time = np.mean(all_times, axis=0)
        print(
            f"Increment {i}\nAverage time for constitutive update {average_time[0]:.2f}s"
        )
        print(f"Average time for global linear solver {average_time[1]:.2f}s\n")

file_results.close()
# -

# JAX AD avoids us to compute explicitly the consistent tangent operator which is particularly nasty for such $\bFe\bFp$ models. We observe that the Newton method converges within a few iterations in general.
