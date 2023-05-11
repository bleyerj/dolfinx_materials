import ufl
import basix
import numpy as np
from dolfinx import fem


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
    fun.vector.array[:] = array.flatten()


class QuadratureMap:
    def __init__(self, mesh, deg, g, material, dim=None):
        # Define mesh and cells
        self.mesh = mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, self.num_cells, dtype=np.int32)
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": deg})

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
        ).flatten()

    def update_flux(self, eval_flux_list, cell_groups):
        g_vals = self.get_gradient_vals()
        flux_vals = np.zeros_like(get_vals(self.flux))
        Ct_vals = np.zeros_like(get_vals(self.jacobian_flatten))
        self.final_state = {
            key: np.zeros_like(get_vals(param))
            for key, param in self.variables.items()
        }
        for eval_flux, cells in zip(eval_flux_list, cell_groups):
            dofs = self._cell_to_dofs(cells)
            g_vals_block = g_vals[dofs, :]
                # old_state = {
                #     key: get_vals(param)[:, dof]
                #     for key, param in self.variables.items()
                # }
            flux_vals[dofs, :], Ct_vals_mat = eval_flux(
                g_vals_block
            )
            Ct_vals[dofs, :] = Ct_vals_mat.flatten()

                # for key, p in self.final_state.items():
                #     p[:, dof] = new_state[key]

        # self.advance()
        update_vals(self.flux, flux_vals)
        update_vals(self.jacobian_flatten, Ct_vals)

    def advance(self):
        self.material.data_manager.update()
        final_state = self.material.get_final_state_dict()
        for key in self.variables.keys():
            update_vals(self.variables[key], final_state[key])
