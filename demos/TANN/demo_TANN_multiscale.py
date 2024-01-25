import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, io
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.material import TannMaterial
from dolfinx_materials.utils import symmetric_tensor_to_vector
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.quadrature_map import QuadratureMap

L = 3.0
N = 8

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[-0.5, -0.5, 0.0], [0.5, 0.5, L]],
    [N, N, 4 * N],
    mesh.CellType.hexahedron,
)

degree = 1
shape = (2,)
V = fem.functionspace(domain, ("P", degree, shape))


def bottom(x):
    return np.isclose(x[2], 0)


def top(x):
    return np.isclose(x[2], L)


bottom_dofs = fem.locate_dofs_geometrical(V, bottom)
top_dofs = fem.locate_dofs_geometrical(V, top)

u_bot = fem.Function(V)
u_top = fem.Function(V)

displ = fem.Constant(domain, np.zeros((3,)))

bcs = [fem.dirichletbc(u_bot, bottom_dofs), fem.dirichletbc(displ, top_dofs, V)]

nisv = 22  # number of internal state variables
material = TannMaterial("demos/TANN/TANN_material", nisv)

u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)
du = ufl.TrialFunction(V)


def strain(u):
    return symmetric_tensor_to_vector(ufl.sym(ufl.grad(u)))


qmap = QuadratureMap(domain, 4, material)
qmap.register_gradient("Strain", strain(u))
sig = qmap.fluxes["Stress"]
Res = ufl.inner(sig, strain(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6
newton.convergence_criterion = "incremental"
newton.report = True


U_max = 2
Nsteps = 20
U_list = np.concatenate(
    (np.linspace(0, U_max, Nsteps + 1)[1:], np.linspace(U_max, 0, Nsteps // 2)[1:])
)

out_file = "TANN.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)

u.vector.set(0.0)
for n, Ui in enumerate(U_list):
    displ.value[0] = Ui

    num_its, converged = problem.solve(newton)
    assert converged

    isv = qmap.project_on("ivars", ("DG", 0))

    with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
        xdmf.write_function(u, n + 1)
        xdmf.write_function(isv, n + 1)
