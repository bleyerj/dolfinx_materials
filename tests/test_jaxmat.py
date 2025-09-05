import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, io

from dolfinx_materials.quadrature_map import QuadratureMap

from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx_materials.solvers import NonlinearMaterialProblem

import jaxmat.materials as jm
from jaxmat.tensors import Tensor
import jax.numpy as jnp
import equinox as eqx

elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)


class VoceHardening(eqx.Module):
    sig0: float
    sigu: float
    b: float

    def __call__(self, p):
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-self.b * p))


hardening = VoceHardening(sig0=350.0, sigu=500.0, b=1e3)

behavior = jm.vonMisesIsotropicHardening(
    elastic_model=elasticity, yield_stress=hardening
)


def get_shape(val):
    if isinstance(val, Tensor):
        return val.array_shape[0]
    else:
        shape = val.shape
        if shape == ():
            return 1
        else:
            return shape[0]


from dolfinx_materials.material import Material
import jax


class DataManager:
    def __init__(self, material, ngauss):
        num_gradients = sum([v for v in material.gradients.values()])
        num_fluxes = sum([v for v in material.fluxes.values()])
        self.K = np.zeros((num_fluxes, num_gradients))
        self.s0 = material.behavior.init_state(ngauss)
        self.s1 = material.behavior.init_state(ngauss)


def replace_elem(key_type, val):
    if issubclass(key_type, Tensor):
        return key_type(array=val)
    else:
        return val


def get_elem(key_type, val):
    if issubclass(key_type, Tensor):
        return val.array
    else:
        return val


def jaxmat_to_dolfinx_state(jaxmat_state):
    print("jaxmat_state", jaxmat_state)
    dolfinx_state = jax.tree.map(lambda x: x.array, jaxmat_state)
    print("dolfinx_state", dolfinx_state)
    return dolfinx_state


def dolfinx_to_jaxmat_state(dolfinx_state, jaxmat_state):
    for key, val in dolfinx_state.items():
        if hasattr(jaxmat_state, key):
            key_type = getattr(jaxmat_state, key).__class__
            jaxmat_state = eqx.tree_at(
                lambda state: getattr(state, key),
                jaxmat_state,
                replace_elem(key_type, val),
            )
        elif hasattr(jaxmat_state.internal, key):
            key_type = getattr(jaxmat_state.internal, key).__class__
            jaxmat_state = eqx.tree_at(
                lambda state: getattr(state.internal, key),
                jaxmat_state,
                replace_elem(key_type, val),
            )
        else:
            raise ValueError(f"Key {key} is missing from material state")
    return jaxmat_state


def as_jaxmat_tensor(tensor_type, array):
    return tensor_type(array=array)


class JAXMaterial(Material):
    def __init__(self, behavior):
        self.behavior = behavior
        self.material_properties = self.behavior.elastic_model.__dict__
        self.batched_constitutive_update = eqx.filter_vmap(
            jax.jacfwd(self.constitutive_update, argnums=0, has_aux=True),
            in_axes=(0, 0, None),
            out_axes=(0, 0),
        )

    def constitutive_update(self, gradients, state, dt):
        tensor_grad = as_jaxmat_tensor(self.data_manager.s0.strain.__class__, gradients)
        stress, new_state = self.behavior.constitutive_update(tensor_grad, state, dt)
        print("Dolfinx_state", jaxmat_to_dolfinx_state(new_state))
        return stress.array, jaxmat_to_dolfinx_state(new_state)

    @property
    def gradients(self):
        return {"strain": 6}

    @property
    def fluxes(self):
        return {"stress": 6}

    @property
    def internal_state_variables(self):
        return {key: get_shape(val) for key, val in behavior.internal.__dict__.items()}

    def set_data_manager(self, ngauss):
        # Setting the material data manager
        self.data_manager = DataManager(self, ngauss)

    def get_initial_state_dict(self):
        return self.data_manager.s0

    def get_final_state_dict(self):
        return self.data_manager.s1

    def set_initial_state_dict(self, state):
        self.data_manager.s0 = eqx.filter_vmap(dolfinx_to_jaxmat_state, in_axes=0)(
            state, self.data_manager.s0
        )
        print("Set intial state dict", self.data_manager.s0)
        # for key, val in state.items():
        #     if hasattr(init_state, key):
        #         key_type = getattr(init_state, key).__class__
        #         print(key, key_type, val)
        #         setattr(init_state, key, key_type(array=val))
        #     elif hasattr(init_state.internal, key):
        #         key_type = getattr(init_state.internal, key).__class__
        #         setattr(init_state.internal, key, key_type(array=val))
        #     else:
        #         raise ValueError(f"Key {key} is missing from material state")

    def integrate(self, gradients, dt=0):
        vectorized_state = self.get_initial_state_dict()
        Ct_array, new_state = self.batched_constitutive_update(
            gradients, vectorized_state, dt
        )
        print(new_state)
        self.data_manager.s1 = new_state
        raise
        return (
            self.data_manager.s1.fluxes,
            self.data_manager.s1.internal_state_variables,
            Ct_array,
        )


material = JAXMaterial(behavior)
material_prop = material.material_properties
print(material_prop)
# print(material.tangent_blocks)
# raise
# from jaxmat.materials import Isotropic

# from dolfinx.cpp.nls.petsc import NewtonSolver  # noqa

order = 1
N = 2

domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("P", order, (2,)))

deg_quad = 2 * (order - 1)


def bottom(x):
    return np.isclose(x[1], 0)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1.0)


V_ux, _ = V.sub(0).collapse()
left_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), left)
right_dofs_ux = fem.locate_dofs_geometrical((V.sub(0), V_ux), right)
V_uy, _ = V.sub(1).collapse()
bottom_dofs_uy = fem.locate_dofs_geometrical((V.sub(1), V_uy), bottom)

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
# u.x.array[:] = np.random.rand(len(u.x.array))


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


qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient(material.gradient_names[0], strain(u))

sig = qmap.fluxes[material.flux_names[0]]
Res = ufl.dot(sig, strain(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)
newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6
newton.max_it = 20


Exx = np.linspace(0, 1e-3, 10)
# if save_fields:
#     file_results = io.XDMFFile(
#         domain.comm,
#         f"{material.name}_results.xdmf",
#         "w",
#     )
#     file_results.write_mesh(domain)
Stress = np.zeros((len(Exx), 6))
for i, exx in enumerate(Exx[1:]):
    uD_x_r.x.array[:] = exx

    converged, _ = problem.solve(newton, False)

    assert converged
    Stress[i + 1, :] = sig.x.array[:6]

    # if save_fields:
    #     for field_name in save_fields:
    #         field = qmap.project_on(field_name, ("DG", 0))
    #         file_results.write_function(field, i)
