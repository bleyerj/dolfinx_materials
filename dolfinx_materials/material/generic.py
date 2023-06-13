import numpy as np
from dolfinx.common import Timer


class Material:
    def __init__(self, **kwargs):
        self.material_properties = self.default_properties()
        self.material_properties.update(kwargs)
        for key, value in self.material_properties.items():
            setattr(self, key, value)

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

    def integrate(self, gradients):
        try:
            vectorized_state = self.get_initial_state_dict()
            flux_array, Ct_array = self.constitutive_update_vectorized(
                gradients, vectorized_state
            )
            self.data_manager.s1.set_item(vectorized_state)

        # with Timer("dx_mat: Python loop constitutive update"):
        #     state = self.data_manager.s0
        #     results = [
        #         self.constitutive_update(g, state[i]) for i, g in enumerate(gradients)
        #     ]
        #     flux_array = np.array([res[0] for res in results])
        #     Ct_array = np.array([res[1] for res in results])
        except AttributeError:
            flux_array = []
            Ct_array = []
            for i, g in enumerate(gradients):
                with Timer("dx_mat: get state"):
                    state = self.data_manager.s0[i]
                with Timer("dx_mat: const update"):
                    flux, Ct = self.constitutive_update(g, state)
                with Timer("dx_mat: appends"):
                    flux_array.append(flux)
                    Ct_array.append(Ct.ravel())

                with Timer("dx_mat: set state"):
                    self.data_manager.s1[i] = state

        return np.array(flux_array), np.array(Ct_array)

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
