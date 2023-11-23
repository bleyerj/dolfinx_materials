import ufl
import basix
import numpy as np
from dolfinx import fem
from .utils import (
    project,
    get_function_space_type,
    create_quadrature_space,
    create_vector_quadrature_space,
    create_tensor_quadrature_space,
    to_mat,
    get_vals,
    update_vals,
)
from dolfinx.common import Timer
from .material import Material
from .quadrature_function import create_quadrature_function, QuadratureExpression
from mpi4py import MPI


def my_dot(a, b):
    if a.ufl_shape == ():
        return a * b
    if b.ufl_shape == ():
        return b * a
    return ufl.dot(a, b)


def mpi_print(s):
    print(f"Rank {MPI.COMM_WORLD.rank}: {s}")


class QuadratureMap:
    def __init__(self, mesh, deg, material, dim=None, cells=None):
        # Define mesh and cells
        self.mesh = mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        if cells is None:
            self.num_cells = map_c.size_local + map_c.num_ghosts
            self.cells = np.arange(0, self.num_cells, dtype=np.int32)
        else:
            self.num_cells = len(cells)
            self.cells = cells
        self.dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": deg})
        self.mesh_dim = self.mesh.geometry.dim

        self.degree = deg

        self.material = material

        self.gradients = {}

        buff = 0
        for block, shape in self.material.tangent_blocks.items():
            buff += np.prod(shape)
        self.WJ = create_vector_quadrature_space(self.mesh, self.degree, buff)
        self.jacobian_flatten = fem.Function(self.WJ)
        self.jacobians = {}
        buff = 0
        for block, shape in self.material.tangent_blocks.items():
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

        self.fluxes = {}
        for name, dim in self.material.fluxes.items():
            self.fluxes.update(
                {name: create_quadrature_function(name, dim, self.mesh, self.degree)}
            )

        self.internal_state_variables = {}
        for name, dim in self.material.internal_state_variables.items():
            self.internal_state_variables.update(
                {name: create_quadrature_function(name, dim, self.mesh, self.degree)}
            )

        self.external_state_variables = {}

        self.set_data_manager(self.cells)

        if self.material.rotation_matrix is not None:
            Wrot = create_tensor_quadrature_space(self.mesh, self.degree, (3, 3))
            self.rotation_func = fem.Function(Wrot)
            self.eval_quadrature(self.material.rotation_matrix, self.rotation_func)

        self.update_material_properties()

        self._initialized = False

    def derivative(self, F, u, du):
        """Computes derivative of non-linear form F"""
        # derivatives of variable u
        tangent_form = ufl.derivative(F, u, du)
        for dy, dx in self.material.tangent_blocks:
            if dy in self.fluxes:
                var_y = self.fluxes[dy]
            elif dy in self.internal_state_variables:
                var_y = self.internal_state_variables[dy]
            else:
                raise ValueError(
                    f"Function '{dy}' to differentiate is not a flux or an internal state variable."
                )

            Tang = self.jacobians[(dy, dx)]
            if dx in self.gradients:
                delta_dx = self.gradients[dx].variation(u, du)
            elif dx in self.external_state_variables:
                delta_dx = self.external_state_variables[dx].variation(u, du)
            else:
                raise ValueError(
                    f"Function '{dx}' to differentiate with is not a gradient or an external state variable."
                )
            tdg = my_dot(Tang, delta_dx)
            tangent_form += ufl.derivative(F, var_y, tdg)
        return tangent_form

    def update_material_properties(self):
        for name, mat_prop in self.material.material_properties.items():
            if isinstance(mat_prop, (int, float, np.ndarray)):
                values = mat_prop
            else:
                fs_type = get_function_space_type(mat_prop)
                Vm = create_quadrature_space(self.mesh, self.degree, *fs_type)
                mat_prop_fun = fem.Function(Vm, name=name)
                self.eval_quadrature(mat_prop, mat_prop_fun)
                values = mat_prop_fun.vector.array
            if values is not None:
                self.material.update_material_property(name, values)

    # def initialize_function(self, fun):
    #     fun.eval(self.cells)
    #     values = fun.function.vector.array

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
        if name in self.material.gradients:
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
            self.material.update_external_state_variable(name, values)

    def set_data_manager(self, cells):
        self.dofs = self._cell_to_dofs(cells)
        self.material.set_data_manager(len(self.dofs))

    @property
    def variables(self):
        return {**self.gradients, **self.fluxes, **self.internal_state_variables}

    @property
    def quadrature_points(self):
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_types[0].name)
        quadrature_points, weights = basix.make_quadrature(basix_celltype, self.degree)
        return quadrature_points

    def eval_quadrature(self, ufl_expr, fem_func):
        expr_expr = fem.Expression(ufl_expr, self.quadrature_points)
        expr_eval = expr_expr.eval(self.cells)
        fem_func.vector.array[:] = expr_eval.flatten()[:]

    def get_gradient_vals(self, gradient, cells):
        gradient.eval(cells)
        return get_vals(gradient.function)[self.dofs, :]

    def _cell_to_dofs(self, cells):
        num_qp = len(self.quadrature_points)
        return (
            np.repeat(num_qp * cells[:, np.newaxis], num_qp, axis=1)
            + np.repeat(np.arange(num_qp)[np.newaxis, :], len(cells), axis=0)
        ).ravel()

    def update_initial_state(self, field_name):
        if field_name in self.fluxes:
            field = self.fluxes[field_name]
        elif field_name in self.internal_state_variables:
            field = self.internal_state_variables[field_name]
        else:
            raise ValueError("Can only initialize a flux or internal state variables.")
        self.material.set_initial_state_dict({field_name: get_vals(field)[self.dofs]})

    def reinitialize_state(self):
        state_flux = {
            key: get_vals(field)[self.dofs] for key, field in self.fluxes.items()
        }
        state_isv = {
            key: get_vals(field)[self.dofs]
            for key, field in self.internal_state_variables.items()
        }
        state_grad = {
            key: self.get_gradient_vals(field, self.cells)
            for key, field in self.gradients.items()
        }
        state = {**state_grad, **state_flux, **state_isv}
        self.material.set_initial_state_dict(state)

    def update(self):
        num_QP = len(self.quadrature_points) * self.num_cells
        if not self._initialized:
            self.reinitialize_state()
            self._initialized = True

        with Timer("External state var update"):
            self.update_external_state_variables()

        with Timer("Eval gradients"):
            grad_vals = []
            # loop over gradients in proper order
            for name in self.material.gradients.keys():
                grad = self.gradients[name]
                grad_vals.append(
                    self.get_gradient_vals(grad, self.cells)
                )  # copy to avoid changing gradient values when rotating
            grad_vals = np.concatenate(grad_vals)

        if self.material.rotation_matrix is not None:
            self.material.rotate_gradients(
                grad_vals.ravel(), self.rotation_func.vector.array
            )

        flux_size = sum(list(self.material.fluxes.values()))
        flux_vals = np.zeros((num_QP, flux_size))
        Ct_vals = np.zeros_like(get_vals(self.jacobian_flatten)[self.dofs])

        flux_vals, isv_vals, Ct_vals = self.material.integrate(grad_vals)

        if self.material.rotation_matrix is not None:
            self.material.rotate_fluxes(
                flux_vals.ravel(), self.rotation_func.vector.array
            )
            self.material.rotate_tangent_operator(
                Ct_vals.ravel(), self.rotation_func.vector.array
            )

        self.update_fluxes(flux_vals)
        self.update_internal_state_variables(isv_vals)
        update_vals(self.jacobian_flatten, Ct_vals, self.cells)

    def update_fluxes(self, flux_vals):
        buff = 0
        for name, dim in self.material.fluxes.items():
            flux = self.fluxes[name]
            update_vals(flux, flux_vals[:, buff : buff + dim], self.cells)
            buff += dim

    def update_internal_state_variables(self, isv_vals):
        buff = 0
        for name, dim in self.material.internal_state_variables.items():
            isv = self.internal_state_variables[name]
            update_vals(isv, isv_vals[:, buff : buff + dim], self.cells)
            buff += dim

    def advance(self):
        self.material.data_manager.update()
        final_state = self.material.get_final_state_dict()
        for key in self.variables.keys():
            if key not in self.gradients:  # update flux and isv but not gradients
                update_vals(self.variables[key], final_state[key], self.cells)

    def project_on(self, key, interp):
        if key not in self.variables:
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

        try:
            shape = ufl.shape(field)
        except ufl.log.UFLValueError:
            shape = field.ufl_shape
            field = field.expression.ufl_expression

        if shape == ():
            V = fem.FunctionSpace(self.mesh, interp)
            projected = fem.Function(V, name=key)
            project(field, projected, self.dx)
            return projected
        else:
            dim = shape[0]
        if dim <= 1:
            V = fem.FunctionSpace(self.mesh, interp)
            projected = fem.Function(V, name=key)
            project(field[0], projected, self.dx)
        else:
            V = fem.VectorFunctionSpace(self.mesh, interp, dim=dim)
            projected = fem.Function(V, name=key)
            project(field, projected, self.dx)
        return projected
