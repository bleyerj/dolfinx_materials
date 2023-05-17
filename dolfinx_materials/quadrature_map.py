import ufl
import basix
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from .utils import (
    project,
    get_function_space_type,
    create_quadrature_space,
    create_scalar_quadrature_space,
    create_vector_quadrature_space,
    create_tensor_quadrature_space,
    to_mat,
    get_vals,
    update_vals,
)
from .material import Material
from .quadrature_function import create_quadrature_function, QuadratureExpression


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

        self.Wf = create_vector_quadrature_space(self.mesh, self.degree, f_dim)
        self.gradients = {}
        self.fluxes = {}

        self.flux = fem.Function(self.Wf)

        buff = 0
        for block, shape in self.material.get_tangent_blocks().items():
            buff += np.prod(shape)
        self.WJ = create_vector_quadrature_space(self.mesh, self.degree, buff)
        self.jacobian_flatten = fem.Function(self.WJ)
        self.jacobians = {}
        buff = 0
        for block, shape in self.material.get_tangent_blocks().items():
            curr_dim = np.prod(shape)
            self.jacobians.update(
                {
                    block: to_mat(
                        [
                            [
                                self.jacobian_flatten[i + shape[1] * j]
                                for i in range(buff, buff + shape[1])
                            ]
                            for j in range(shape[0])
                        ]
                    )
                }
            )
            buff += curr_dim

        self.variables = {}
        for key, dim in self.material.get_variables().items():
            self._add_variable(dim, key)

        for name, dim in self.material.get_fluxes().items():
            self.fluxes.update(
                {key: create_quadrature_function(name, dim, self.mesh, self.degree)}
            )

        self.external_state_variables = {}

        self.set_data_manager(self.cells)

        if self.material.rotation_matrix is not None:
            Wrot = create_tensor_quadrature_space(self.mesh, self.degree, (3, 3))
            self.rotation_func = fem.Function(Wrot)
            self.eval_quadrature(self.material.rotation_matrix, self.rotation_func)

        self.update_material_properties()

    def derivative(self, F, u, du):
        """Computes derivative of residual"""
        # derivatives of variable u
        tangent_form = ufl.derivative(F, u, du)
        variables = {**self.variables, **self.external_state_variables}
        # derivatives of fluxes
        # for fname, fdim in self.fluxes.items():
        fname = list(self.material.get_fluxes().keys())[0]
        tangent_blocks = [
            dx for dy, dx in self.material.get_tangent_blocks().keys() if dy == fname
        ]
        for dx in tangent_blocks:
            if dx in self.material.get_gradients():
                Tang = self.jacobians[(fname, dx)]
                delta_dx = self.gradients[dx].variation(u, du)
                # if len(ufl.shape(tdg)) > 0:
                #     if ufl.shape(tdg)[0] == 1:
                #         tdg = tdg[0]
            elif dx in self.external_state_variables:
                Tang = self.jacobians[(fname, dx)]
                delta_dx = self.external_state_variables[dx].variation(u, du)
            tdg = Tang * delta_dx
            tangent_form += ufl.replace(F, {self.fluxes[fname]: tdg})
        # derivatives of internal state variables
        # for s in self.state_variables["internal"].values():
        #     for t, dg in zip(s.tangent_blocks.values(), s.variables):
        #         tdg = t * dg.variation(self.du)
        #         if len(shape(t)) > 0 and shape(t)[0] == 1:
        #             tdg = tdg[0]
        #         self.tangent_form += derivative(self.residual, s.function, tdg)
        return tangent_form

    def update_material_properties(self):
        for name, mat_prop in self.material.material_properties.items():
            if isinstance(mat_prop, (int, float, np.ndarray)):
                values = mat_prop
            elif isinstance(mat_prop, Material) or callable(mat_prop):  # FIXME
                values = None
            else:
                fs_type = get_function_space_type(mat_prop)
                Vm = create_quadrature_space(self.mesh, self.degree, *fs_type)
                mat_prop_fun = fem.Function(Vm, name=name)
                self.eval_quadrature(mat_prop, mat_prop_fun)
                values = mat_prop_fun.vector.array
            if values is not None:
                self.material.update_material_property(name, values)

    def register_external_state_variable(self, name, ext_state_var):
        """Registers a dolfinx object as an external state variable.

        Parameters
        ----------
        name : str
            Name of the variable
        ext_state_var : float, np.ndarray, fem.Function
            External state variable
        """
        if isinstance(ext_state_var, (int, float, np.ndarray)):
            ext_state_var = fem.Constant(self.mesh, ext_state_var)
        state_var = QuadratureExpression(
            name,
            fem.Expression(ext_state_var, self.quadrature_points),
            self.mesh,
            self.degree,
        )
        self.external_state_variables.update({name: state_var})

    def register_gradient(self, name, gradient):
        if name in self.material.get_gradients():
            grad = QuadratureExpression(
                name,
                fem.Expression(gradient, self.quadrature_points),
                self.mesh,
                self.degree,
            )
            self.gradients.update({name: grad})
        else:
            raise ValueError(
                f"Gradient '{name}' is not available from the material law."
            )

    def update_external_state_variables(self):
        for name, esv in self.external_state_variables.items():
            esv.eval(self.cells)
            values = esv.function.vector.array
            # if isinstance(esv, (int, float, np.ndarray)):
            #     values = esv
            # else:
            #     fs_type = get_function_space_type(esv)
            #     Vm = create_quadrature_space(self.mesh, self.degree, *fs_type)
            #     esv_fun = fem.Function(Vm, name=name)
            #     self.eval_quadrature(esv, esv_fun)
            #     values = esv_fun.vector.array
            self.material.update_external_state_variable(name, values)

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

    @property
    def quadrature_points(self):
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
        quadrature_points, weights = basix.make_quadrature(basix_celltype, self.degree)
        return quadrature_points

    def eval_quadrature(self, ufl_expr, fem_func):
        expr_expr = fem.Expression(ufl_expr, self.quadrature_points)
        expr_eval = expr_expr.eval(self.cells)
        fem_func.vector.array[:] = expr_eval.flatten()[:]

    def get_gradient_vals(self, gradient, cells):
        gradient.eval(cells)
        return get_vals(gradient.function)

    def update(self, cells=None):
        if cells is None:
            self.update_flux([self.material.integrate], [self.cells])
        else:  # FIXME
            self.update_flux(self.material.integrate, cells)

    def _cell_to_dofs(self, cells):
        num_qp = len(self.quadrature_points)
        return (
            np.repeat(num_qp * cells[:, np.newaxis], num_qp, axis=1)
            + np.repeat(np.arange(num_qp)[np.newaxis, :], len(cells), axis=0)
        ).ravel()

    def update_flux(self, eval_flux_list, cell_groups):
        self.update_external_state_variables()

        grad_vals = []
        # loop over gradients in proper order
        for name in self.material.get_gradients().keys():
            grad = self.gradients[name]
            grad_vals.append(
                self.get_gradient_vals(grad, self.cells)
            )  # copy to avoid changing gradient values when rotating
        grad_vals = np.concatenate(grad_vals)

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
        self.update_fluxes(flux_vals)
        update_vals(self.jacobian_flatten, Ct_vals)

    def update_fluxes(self, flux_vals):
        buff = 0
        for name, dim in self.material.get_fluxes().items():
            flux = self.fluxes[name]
            update_vals(flux, flux_vals[:, buff : buff + dim])
            buff += dim

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
