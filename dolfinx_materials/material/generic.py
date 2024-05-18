import numpy as np
from dolfinx.common import Timer
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision


def tangent_AD(compute_stress_method):
    from functools import wraps, partial

    @wraps(compute_stress_method)
    def wrapper(self, *args):
        compute_stress = partial(compute_stress_method, self)
        return jax.jacfwd(compute_stress, argnums=0, has_aux=True)(*args)

    return wrapper


class Material:
    def __init__(self, **kwargs):
        self.material_properties = self.default_properties()
        self.material_properties.update(kwargs)
        for key, value in self.material_properties.items():
            setattr(self, key, value)

        self.batched_constitutive_update = jax.jit(
            jax.vmap(self.constitutive_update, in_axes=(0, 0, None))
        )

    def update_material_property(self, key, value):
        setattr(self, key, value)

    def default_properties(self):
        return {}

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def rotation_matrix(self):
        return None

    @property
    def gradients(self):
        return {"Strain": 6}

    @property
    def fluxes(self):
        return {"Stress": 6}

    @property
    def tangent_blocks(self):
        return {
            (kf, kg): (vf, vg)
            for (kf, vf), (kg, vg) in zip(self.fluxes.items(), self.gradients.items())
        }

    @property
    def internal_state_variables(self):
        return {}

    @property
    def variables(self):
        return {
            **self.gradients,
            **self.fluxes,
            **self.internal_state_variables,
        }

    @property
    def gradient_names(self):
        return list(self.gradients.keys())

    @property
    def flux_names(self):
        return list(self.fluxes.keys())

    @property
    def internal_state_variable_names(self):
        return list(self.internal_state_variables.keys())

    def set_data_manager(self, ngauss):
        # Setting the material data manager
        self.data_manager = DataManager(self, ngauss)

    def integrate(self, gradients, dt=0):
        vectorized_state = self.get_initial_state_dict()
        Ct_array, new_state = self.batched_constitutive_update(
            gradients, vectorized_state, dt
        )
        new_sig = new_state["Stress"]
        self.data_manager.s1.set_item(new_state)
        self.data_manager.s1.set_item({"Stress": new_sig})

        return (
            self.data_manager.s1.fluxes,
            self.data_manager.s1.internal_state_variables,
            Ct_array,
        )

    def constitutive_update(self):
        pass

    def get_initial_state_dict(self):
        return self.data_manager.s0[:]

    def get_final_state_dict(self):
        return self.data_manager.s1[:]

    def set_initial_state_dict(self, state):
        return self.data_manager.s0.set_item(state)


class DataManager:
    def __init__(self, behaviour, ngauss):
        num_gradients = sum([v for v in behaviour.gradients.values()])
        num_fluxes = sum([v for v in behaviour.fluxes.values()])
        self.K = np.zeros((num_fluxes, num_gradients))
        self.s0 = MaterialStateManager(behaviour, ngauss)
        self.s1 = MaterialStateManager(behaviour, ngauss)

    def update(self):
        self.s0.update(self.s1)

    def revert(self):
        self.s1.update(self.s0)


class MaterialStateManager:
    def __init__(self, behaviour, ngauss):
        self._behaviour = behaviour
        self.n = ngauss
        self.gradients_stride = [v for v in self._behaviour.gradients.values()]
        self.fluxes_stride = [v for v in self._behaviour.fluxes.values()]
        self.internal_state_variables_stride = [
            v for v in self._behaviour.internal_state_variables.values()
        ]
        self.gradients = np.zeros((ngauss, sum(self.gradients_stride)))
        self.fluxes = np.zeros((ngauss, sum(self.fluxes_stride)))
        self.internal_state_variables = np.zeros(
            (ngauss, sum(self.internal_state_variables_stride))
        )

    def update(self, other):
        self.gradients = np.copy(other.gradients)
        self.fluxes = np.copy(other.fluxes)
        self.internal_state_variables = np.copy(other.internal_state_variables)

    def get_flux_index(self, name):
        index = self._behaviour.flux_names.index(name)
        pos = np.arange(index, index + self._behaviour.fluxes[name])
        return pos

    def get_gradient_index(self, name):
        index = self._behaviour.gradient_names.index(name)
        pos = np.arange(index, index + self._behaviour.gradients[name])
        return pos

    def get_internal_state_variable_index(self, name):
        index = self._behaviour.internal_state_variable_names.index(name)
        pos = np.arange(index, index + self._behaviour.internal_state_variables[name])
        return pos

    def __getitem__(self, i):
        state = {}
        for key, value in self._behaviour.gradients.items():
            pos = self.get_gradient_index(key)
            state.update({key: self.gradients[i, pos]})
        for key, value in self._behaviour.fluxes.items():
            pos = self.get_flux_index(key)
            state.update({key: self.fluxes[i, pos]})
        for key, value in self._behaviour.internal_state_variables.items():
            pos = self.get_internal_state_variable_index(key)
            state.update({key: self.internal_state_variables[i, pos]})
        return state

    def set_item(self, state, indices=None):
        if indices is None:
            indices = np.arange(self.n)
        state_copy = state.copy()
        for key, value in state.items():
            if key in self._behaviour.gradients:
                pos = self.get_gradient_index(key)
                self.gradients[np.ix_(indices, pos)] = value
                state_copy.pop(key)
            if key in self._behaviour.fluxes:
                pos = self.get_flux_index(key)
                self.fluxes[np.ix_(indices, pos)] = value
                state_copy.pop(key)
            if key in self._behaviour.internal_state_variables:
                pos = self.get_internal_state_variable_index(key)
                self.internal_state_variables[np.ix_(indices, pos)] = value
                state_copy.pop(key)
        assert (
            len(state_copy) == 0
        ), "Material state contains unknown field to update with."

    def __setitem__(self, i, state):
        self.set_item(state, [i])
