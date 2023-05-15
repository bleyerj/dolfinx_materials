import ufl
import basix
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from .utils import project, get_function_space_type

function_space_dict = {
    "scalar": ufl.FiniteElement,
    "vector": ufl.VectorElement,
    "tensor": ufl.TensorElement,
}


def create_quadrature_space(mesh, degree, type, shape):
    We = function_space_dict[type](
        "Quadrature", mesh.ufl_cell(), degree, shape, quad_scheme="default"
    )
    return fem.FunctionSpace(mesh, We)


def create_scalar_quadrature_space(mesh, degree):
    return create_vector_quadrature_space(mesh, degree, 1)


def create_vector_quadrature_space(mesh, degree, dim):
    if dim > 0:
        We = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            degree=degree,
            dim=dim,
            quad_scheme="default",
        )
        return fem.FunctionSpace(mesh, We)
    else:
        raise ValueError("Vector dimension should be at least 1.")


def create_tensor_quadrature_space(mesh, degree, shape):
    We = ufl.TensorElement(
        "Quadrature",
        mesh.ufl_cell(),
        degree=degree,
        shape=shape,
        quad_scheme="default",
    )

    return fem.FunctionSpace(mesh, We)


def get_vals(fun):
    dim = len(fun)
    return fun.vector.array.reshape((-1, dim))


def update_vals(fun, array):
    fun.vector.array[:] = array.ravel()


class QuadratureMap:
    def __init__(self, mesh, deg, g, material, dim=None):
        # Define mesh and cells
        self.mesh = mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, self.num_cells, dtype=np.int32)
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": deg})
        self.mesh_dim = self.mesh.geometry.dim

        self.g_expr = g
        if len(ufl.shape(g)) == 1:
            g_dim = ufl.shape(g)[0]
        else:
            raise NotImplementedError("Only vector expressions are supported.")
        if dim is None:
            f_dim = g_dim
        else:
            f_dim = dim
        self.dim = (f_dim, g_dim)
        self.degree = deg

        self.material = material

        self.Wg = create_vector_quadrature_space(self.mesh, self.degree, g_dim)
        self.Wf = create_vector_quadrature_space(self.mesh, self.degree, f_dim)
        self.WJ = create_vector_quadrature_space(self.mesh, self.degree, g_dim * f_dim)
        self.gradient = fem.Function(self.Wg)
        self.flux = fem.Function(self.Wf)
        self.jacobian_flatten = fem.Function(self.WJ)
        self.jacobian = ufl.as_matrix(
            [
                [self.jacobian_flatten[i + g_dim * j] for i in range(g_dim)]
                for j in range(f_dim)
            ]
        )
        self.variables = {}
        for key, dim in self.material.get_variables().items():
            self._add_variable(dim, key)

        self.set_data_manager(self.cells)

        if self.material.rotation_matrix is not None:
            Wrot = create_tensor_quadrature_space(self.mesh, self.degree, (3, 3))
            self.rotation_func = fem.Function(Wrot)
            self.eval_quadrature(self.material.rotation_matrix, self.rotation_func)

        self.update_material_properties()

    def update_material_properties(self):
        for name, mat_prop in self.material.material_properties.items():
            if isinstance(mat_prop, (fem.Constant, fem.Expression)):
                fs_type = get_function_space_type(mat_prop)
                Vm = create_quadrature_space(self.mesh, self.degree, *fs_type)
                mat_prop_fun = fem.Function(Vm, name=name)
                mat_prop_fun.interpolate(mat_prop)
                values = mat_prop_fun.vector.array
            else:
                values = mat_prop
            self.material.update_material_property(name, values)

    def set_data_manager(self, cells):
        dofs = self._cell_to_dofs(cells)
        self.material.set_data_manager(len(dofs))

    def _add_variable(self, dim=None, name=None):
        if dim is None:
            W = create_scalar_quadrature_space(self.mesh, self.degree)
        else:
            W = create_vector_quadrature_space(self.mesh, self.degree, dim)
        fun = fem.Function(W, name=name)
        self.variables.update({fun.name: fun})

    def get_quadrature_points(self):
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
        quadrature_points, weights = basix.make_quadrature(basix_celltype, self.degree)
        return quadrature_points

    def eval_quadrature(self, ufl_expr, fem_func):
        expr_expr = fem.Expression(ufl_expr, self.get_quadrature_points())
        expr_eval = expr_expr.eval(self.cells)
        fem_func.vector.array[:] = expr_eval.flatten()[:]

    def eval_gradient(self):
        self.eval_quadrature(self.g_expr, self.gradient)

    def get_gradient_vals(self):
        self.eval_gradient()
        return get_vals(self.gradient)

    def update(self, cells=None):
        if cells is None:
            self.update_flux([self.material.integrate], [self.cells])
        else:  # FIXME
            self.update_flux(self.material.integrate, cells)

    def _cell_to_dofs(self, cells):
        num_qp = len(self.get_quadrature_points())
        return (
            np.repeat(num_qp * cells[:, np.newaxis], num_qp, axis=1)
            + np.repeat(np.arange(num_qp)[np.newaxis, :], len(cells), axis=0)
        ).ravel()

    def update_flux(self, eval_flux_list, cell_groups):
        grad_vals = np.copy(
            self.get_gradient_vals()
        )  # copy to avoid changing gradient values when rotating
        if self.material.rotation_matrix is not None:
            self.material.rotate_gradients(
                grad_vals.ravel(), self.rotation_func.vector.array
            )

        flux_vals = np.zeros_like(get_vals(self.flux))
        Ct_vals = np.zeros_like(get_vals(self.jacobian_flatten))
        self.final_state = {
            key: np.zeros_like(get_vals(param)) for key, param in self.variables.items()
        }
        for eval_flux, cells in zip(eval_flux_list, cell_groups):
            dofs = self._cell_to_dofs(cells)
            grad_vals_block = grad_vals[dofs, :]
            flux_vals[dofs, :], Ct_vals_mat = eval_flux(grad_vals_block)
            Ct_vals[dofs, :] = Ct_vals_mat

        if self.material.rotation_matrix is not None:
            self.material.rotate_fluxes(
                flux_vals.ravel(), self.rotation_func.vector.array
            )
            self.material.rotate_tangent_operator(
                Ct_vals.ravel(), self.rotation_func.vector.array
            )
        update_vals(self.flux, flux_vals)
        update_vals(self.jacobian_flatten, Ct_vals)

    def advance(self):
        self.material.data_manager.update()
        final_state = self.material.get_final_state_dict()
        for key in self.variables.keys():
            if (
                key not in self.material.get_gradients().keys()
            ):  # update flux and isv but not gradients
                update_vals(self.variables[key], final_state[key])

    def project_on(self, key, interp):
        if key not in self.variables.keys():
            collected = [
                v[0]
                for k, v in self.variables.items()
                if k.startswith(key) and len(v) == 1
            ]
            if len(collected) == 0:
                raise ValueError(f"Field '{key}' has not been found.")
            field = ufl.as_vector(collected)
        else:
            field = self.variables[key]
        dim = ufl.shape(field)[0]
        if dim <= 1:
            V = fem.FunctionSpace(self.mesh, interp)
            projected = fem.Function(V, name=key)
            project(field[0], projected, self.dx)
        else:
            V = fem.VectorFunctionSpace(self.mesh, interp, dim=dim)
            projected = fem.Function(V, name=key)
            project(field, projected, self.dx)
        return projected
