import ufl
import basix
import numpy as np
from dolfinx import fem
from .utils import (
    project,
    create_quadrature_functionspace,
    to_mat,
    get_vals,
    update_vals,
)
from dolfinx.common import Timer
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
    """Abstract data structure to handle a user-defined material in a form.

    Attributes
    ----------
    gradients: dict
        dictionary of gradients represented by Quadrature functions
    fluxes: dict
        dictionary of fluxes represented by Quadrature functions
    internal_state_variables: dict
        dictionary of internal state variables represented by Quadrature functions
    external_state_variables: dict
        dictionary of internal state variables represented by Quadrature functions
    dx: ufl.Measure
        ufl.Measure defined with consistent quadrature degree
    """

    def __init__(self, mesh, deg, material, cells=None):
        """
        Parameters
        ----------
        mesh : dolfinx.mesh
            The underlying Mesh object
        deg : int
            Degree of quadrature space
        material : dolfinx_materials.Material
            A user-defined material object
        cells : list, np.ndarray
            List of cells affected by the material. If None (default), acts on all cells of the domain.
        """
        # Define mesh and cells
        self.mesh = mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
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
        self.WJ = create_quadrature_functionspace(self.mesh, self.degree, buff)
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

        Wrot = create_quadrature_functionspace(self.mesh, self.degree, (3, 3))
        self.rotation_func = fem.Function(Wrot)
        if self.material.rotation_matrix is not None:
            self.update_material_rotation_matrix()

        self.update_material_properties()

        self._initialized = False

    def derivative(self, F, u, du):
        """Computes derivative of non-linear form F"""
        # standard UFL derivatives of variable u
        tangent_form = ufl.derivative(F, u, du)
        # additional contributions due to tangent blocks
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
        """Update material properties from provided values."""
        for name, mat_prop in self.material.material_properties.items():
            if isinstance(mat_prop, (int, float, np.ndarray)):
                values = mat_prop
            else:
                shape = mat_prop.ufl_shape
                Vm = create_quadrature_functionspace(self.mesh, self.degree, shape)
                mat_prop_fun = fem.Function(Vm, name=name)
                self.eval_quadrature(mat_prop, mat_prop_fun)
                values = mat_prop_fun.x.array
            if values is not None:
                self.material.update_material_property(name, values)

    def register_external_state_variable(self, name, ext_state_var):
        """Registers a UFL expression as an external state variable.

        Parameters
        ----------
        name : str
            Name of the external state variable field
        ext_state_var : float, np.ndarray, UFL expression
            Constant or UFL expression to register, e.g. fem.Function(V, name="Temperature")
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
        state_var.eval(self.cells)
        values = state_var.function.x.array
        self.material.initialize_external_state_variable(name, values)

    def register_gradient(self, name, gradient):
        """Registers a UFL expression as a gradient.

        Parameters
        ----------
        name : str
            Name of the gradient field
        gradient : UFL expression
            UFL expression to register, e.g. ufl.grad(u)
        """
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
        """Update material external state variables with dolfinx values."""
        for name, esv in self.external_state_variables.items():
            esv.eval(self.cells)
            values = esv.function.x.array
            self.material.update_external_state_variable(name, values)

    def update_material_rotation_matrix(self):
        """Update material rotation matrix with dolfinx values."""
        self.eval_quadrature(self.material.rotation_matrix, self.rotation_func)

    def set_data_manager(self, cells):
        self.dofs = self._cell_to_dofs(cells)
        self.material.set_data_manager(len(self.dofs))

    @property
    def variables(self):
        return {**self.gradients, **self.fluxes, **self.internal_state_variables}

    @property
    def quadrature_points(self):
        basix_celltype = getattr(basix.CellType, self.mesh.topology.cell_type.name)
        quadrature_points, weights = basix.make_quadrature(basix_celltype, self.degree)
        return quadrature_points

    def eval_quadrature(self, ufl_expr, fem_func):
        """Evaluates an expression at quadrature points and updates the corresponding function."""
        expr_expr = fem.Expression(ufl_expr, self.quadrature_points)
        expr_eval = expr_expr.eval(self.mesh, self.cells)
        fem_func.x.array[:] = expr_eval.flatten()[:]

    def get_gradient_vals(self, gradient, cells):
        gradient.eval(cells)
        return get_vals(gradient.function)[self.dofs, :]

    def _cell_to_dofs(self, cells):
        num_qp = len(self.quadrature_points)
        return (
            np.repeat(num_qp * cells[:, np.newaxis], num_qp, axis=1)
            + np.repeat(np.arange(num_qp)[np.newaxis, :], len(cells), axis=0)
        ).ravel()

    def update_initial_state(self, field_name, value=None):
        """Update a material field with corresponding dolfinx object."""
        if field_name in self.fluxes:
            field = self.fluxes[field_name]
        elif field_name in self.internal_state_variables:
            field = self.internal_state_variables[field_name]
        else:
            raise ValueError("Can only initialize a flux or internal state variables.")
        # if a value is provided we update the field with t
        values = get_vals(field)[self.dofs]
        if isinstance(value, (int, float, np.ndarray)):
            values = np.full_like(values, value)
            update_vals(field, values, self.cells)
        elif value is not None:
            self.eval_quadrature(value, field)
            values = get_vals(field)[self.dofs]
        self.material.set_initial_state_dict({field_name: values})

    def initialize_state(self):
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
        self._initialized = True

    def update(self):
        """Perform constitutive update call."""
        num_QP = len(self.quadrature_points) * self.num_cells
        if not self._initialized:
            self.initialize_state()

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
            grad_vals = np.concatenate(grad_vals, axis=1)

        if self.material.rotation_matrix is not None:
            self.material.rotate_gradients(
                grad_vals.ravel(), self.rotation_func.x.array
            )

        flux_size = sum(list(self.material.fluxes.values()))
        flux_vals = np.zeros((num_QP, flux_size))
        Ct_vals = np.zeros_like(get_vals(self.jacobian_flatten)[self.dofs])

        # material integration
        # print("Grads", grad_vals)
        flux_vals, isv_vals, Ct_vals = self.material.integrate(grad_vals)
        # print("Fluxes", flux_vals)
        assert not (np.any(np.isnan(flux_vals)))
        assert not (np.any(np.isnan(isv_vals)))
        assert not (np.any(np.isnan(Ct_vals)))

        if self.material.rotation_matrix is not None:
            self.material.rotate_fluxes(flux_vals.ravel(), self.rotation_func.x.array)
            self.material.rotate_tangent_operator(
                Ct_vals.ravel(), self.rotation_func.x.array
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
        """
        Advance in time by copying current state dictionary in old state.
        Must be used only after having converged globally.
        """
        self.material.data_manager.update()
        final_state = self.material.get_final_state_dict()
        for key in self.variables.keys():
            if key not in self.gradients:  # update flux and isv but not gradients
                update_vals(self.variables[key], final_state[key], self.cells)

    def project_on(self, name, interp, entity_maps=None):
        """
        Computes a projection onto standard FE space.

        Parameters
        ----------
        name : str
            Name of the field to project
        interp : tuple
            Tuple of interpolation space info to project on, e.g. ("DG", 0). Shape is automatically deduced.
        entity_maps: dict
            Entity maps in case of mixed subdomains
        """
        if name not in self.variables:
            collected = [
                v[0]
                for k, v in self.variables.items()
                if k.startswith(name) and len(v) == 1
            ]
            if len(collected) == 0:
                raise ValueError(f"Field '{name}' has not been found.")
            field = ufl.as_vector(collected)
        else:
            field = self.variables[name]

        try:
            shape = ufl.shape(field)
        except ufl.log.UFLValueError:
            shape = field.ufl_shape
            field = field.expression.ufl_expression

        V = fem.functionspace(self.mesh, interp + (shape,))
        projected = fem.Function(V, name=name)
        project(field, projected, self.dx, entity_maps=entity_maps)
        return projected
