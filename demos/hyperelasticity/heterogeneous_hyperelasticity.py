import numpy as np
import ufl
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
from dolfinx import fem, mesh, io, la, log
from dolfinx.common import list_timings, TimingType
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import (
    symmetric_tensor_to_vector,
    nonsymmetric_tensor_to_vector,
)

Nx, order = 20, 1
domain = mesh.create_unit_cube(
    MPI.COMM_WORLD,
    Nx,
    Nx,
    Nx,
    cell_type=mesh.CellType.hexahedron,
    ghost_mode=mesh.GhostMode.none,
)
gdim = domain.topology.dim

V = fem.functionspace(domain, ("P", order, (3,)))
deg_quad = 2 * order

# print(V.dofmap.list[[0, 1]])

E = 1e9

material = MFrontMaterial(
    "dolfinx_materials/mfront_materials/src/libBehaviour.so",
    "Ogden",
)

N = 10
# Exx = np.linspace(0, 2e-2, N + 1)
Exx = np.concatenate(
    (np.linspace(1e-6, 8e-2, N + 1), np.linspace(8e-2, 20e-2, N + 1)[1:])
)


centers = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
radius = 0.4


def inclusion(x):
    markers = []
    for center in centers:
        print(center)
        markers.append(
            (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 + (x[2] - center[2]) ** 2
            <= radius**2
        )
    marker = np.any(np.vstack(markers), axis=0)
    return marker


def bottom(x):
    return np.isclose(x[1], 0)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1.0)


def border(x):
    return np.logical_and(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)), inclusion(x)
    )


tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
# dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
dofs = fem.locate_dofs_geometrical(V, border)

x = ufl.SpatialCoordinate(domain)
exx = fem.Constant(domain, 0.0)
Fmacro = ufl.as_matrix([[exx, 0, 0], [0, 0, 0], [0, 0, 0]])
u_expr = fem.Expression(ufl.dot(Fmacro, x), V.element.interpolation_points())

uD = fem.Function(V)
bcs = [fem.dirichletbc(uD, dofs)]

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u = fem.Function(V, name="Displacement")


def build_nullspace(V):
    """Build PETSc nullspace for 3D elasticity"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    length1 = length0 + V.dofmap.index_map.num_ghosts
    basis = [np.zeros(bs * length1, dtype=PETSc.ScalarType) for i in range(6)]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.array.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.array.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    basis[3][dofs[0]] = -x1
    basis[3][dofs[1]] = x0
    basis[4][dofs[0]] = x2
    basis[4][dofs[2]] = -x0
    basis[5][dofs[2]] = x1
    basis[5][dofs[1]] = -x2

    # Create PETSc Vec objects (excluding ghosts) and normalise
    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)
        for x in basis
    ]
    la.orthonormalize(basis_petsc)
    assert la.is_orthonormal(basis_petsc)

    # Create and return a PETSc nullspace
    return PETSc.NullSpace().create(vectors=basis_petsc)


def strain(u):
    return symmetric_tensor_to_vector(ufl.sym(ufl.grad(u)))


def F(u):
    return nonsymmetric_tensor_to_vector(ufl.Identity(gdim) + ufl.grad(u))


def dF(u):
    return nonsymmetric_tensor_to_vector(ufl.grad(u))


def matrix(x):
    return np.logical_not(inclusion(x))


V0 = fem.functionspace(domain, ("DG", 0))
kappa = fem.Function(V0)
cells_incl = mesh.locate_entities(domain, gdim, inclusion)
cells_matrix = mesh.locate_entities(domain, gdim, matrix)
kappa.vector.array[:] = 0.0
kappa.x.array[cells_incl] = np.full_like(cells_incl, 1, dtype=ScalarType)

num_cells_local = domain.topology.index_map(gdim).size_local
marker = 2 * np.ones(num_cells_local, dtype=np.int32)
cells_0 = cells_incl[cells_incl < num_cells_local]
marker[cells_0] = 1
cell_tag = mesh.meshtags(domain, gdim, np.arange(num_cells_local), marker)


print(cell_tag.values)
cells_matrix = cell_tag.find(2)
qmap = QuadratureMap(domain, deg_quad, material, cells=cells_matrix)
qmap.register_gradient("DeformationGradient", F(u))
dx = qmap.dx(subdomain_data=cell_tag)

print(qmap.internal_state_variables)
sig = qmap.fluxes["FirstPiolaKirchhoffStress"]
Res = 1e-6 * (ufl.dot(sig, dF(v)) * dx(2) + E * ufl.dot(strain(u), strain(v)) * dx(1))
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-4
newton.convergence_criterion = "incremental"
newton.report = True
log.set_log_level(log.LogLevel.ERROR)

# Set solver options
ksp = newton.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1e-8
opts[f"{option_prefix}pc_type"] = "gamg"

# Use Chebyshev smoothing for multigrid
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"] = "jacobi"

# # Improve estimate of eigenvalues for Chebyshev smoothing
opts["mg_levels_esteig_ksp_type"] = "gmres"
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

ksp.setFromOptions()

file_results = io.XDMFFile(
    domain.comm,
    f"{material.name}_results.xdmf",
    "w",
)
file_results.write_mesh(domain)
file_results.write_function(kappa)
Sxx = np.zeros_like(Exx)
for i, exx_v in enumerate(Exx):
    exx.value = exx_v
    uD.interpolate(u_expr)

    print(f"Increment {i}")

    converged, it = problem.solve(newton)

    # converged, it = snes.solve(snes_solver)
    # assert snes_solver.getConvergedReason() > 0

    Sxx[i] = sig.vector.array[0]

    # p = qmap.project_on("EquivalentPlasticStrain", ("DG", 0))
    # e = qmap.project_on("Strain", ("DG", 0))
    file_results.write_function(u, i)


list_timings(domain.comm, [TimingType.wall, TimingType.user])
