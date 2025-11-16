import numpy as np
from dolfinx.common import Timer
import warnings
from dolfinx_materials import PerformanceWarning

# FIXME: Warnings do not seem to be working at the moment
warnings.simplefilter("once", PerformanceWarning)


def _vmap(fn, in_axes=0, out_axes=0):
    """
    Vectorizes the given function fn, applying it along the specified axes.

    Parameters:
    fn : function
        The function to be vectorized.
    in_axes : int or tuple of ints, optional
        Specifies which axis to map over for each input. If a single int is provided,
        that axis is used for all inputs. If a tuple is provided, each entry corresponds
        to a different input. Default is 0.
    out_axes : int or tuple of ints, optional
        Specifies the axis along which outputs should be stacked. If a single int is
        provided, that axis is used for all outputs. If a tuple is provided, each entry
        corresponds to a different output. Default is 0.

    Returns:
    vectorized_fn : function
        The vectorized version of the input function.
    """

    def vectorized_fn(*args):
        def moveaxis_if_array(x, axis):
            if isinstance(x, np.ndarray):
                return np.moveaxis(x, axis, 0)
            return x

        def slice_data(x, index):
            if isinstance(x, np.ndarray):
                return x[index]
            elif isinstance(x, dict):
                return {key: value[index] for key, value in x.items()}
            return x

        def stack_outputs(outputs, out_axes):
            if isinstance(outputs[0], np.ndarray) or np.isscalar(outputs[0]):
                return np.stack(outputs, axis=out_axes)
            elif isinstance(outputs[0], dict):
                return {
                    key: np.stack([output[key] for output in outputs], axis=out_axes)
                    for key in outputs[0].keys()
                }
            return outputs

        # Handle the case where in_axes is a single int or tuple of ints
        if isinstance(in_axes, int):
            in_axes_tuple = (in_axes,) * len(args)
        else:
            in_axes_tuple = in_axes

        # Move the in_axes to the first axis for each input
        moved_inputs = [
            moveaxis_if_array(arg, axis) for arg, axis in zip(args, in_axes_tuple)
        ]

        # Determine the size of the first dimension (after axis move) to loop over
        loop_size = (
            moved_inputs[0].shape[0]
            if isinstance(moved_inputs[0], np.ndarray)
            else len(next(iter(moved_inputs[0].values())))
        )

        # Apply the function to each slice of the inputs along the first axis
        warnings.warn(
            "Looping over all quadrature points. This might be long...",
            PerformanceWarning,
        )
        results = [
            fn(*[slice_data(arg, i) for arg in moved_inputs]) for i in range(loop_size)
        ]

        # Handle the case where out_axes is a single int or tuple of ints
        if isinstance(out_axes, int):
            out_axes_tuple = (out_axes,)
        else:
            out_axes_tuple = out_axes

        # Stack the results along the specified out_axes
        if isinstance(results[0], tuple):
            # If the function returns multiple outputs, handle each separately
            stacked_result = tuple(
                stack_outputs([res[i] for res in results], out_axes_tuple[i])
                for i in range(len(results[0]))
            )
        else:
            # If the function returns a single output
            stacked_result = stack_outputs(results, out_axes)

        return stacked_result

    return vectorized_fn


class Material:
    """
    This class is used to define a material behavior implemented in pure Python.
    This will be extremely slow if looping over all quadrature points.
    """

    def __init__(self, **kwargs):
        self.material_properties = self.default_properties()
        self.material_properties.update(kwargs)
        for key, value in self.material_properties.items():
            setattr(self, key, value)

        self.batched_constitutive_update = _vmap(
            self.constitutive_update, in_axes=(0, 0, None), out_axes=(0, 0)
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

        self.data_manager.s1.set_item(new_state)

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
        self.gradients_size = [max(1, v) for v in self._behaviour.gradients.values()]
        self.fluxes_size = [max(1, v) for v in self._behaviour.fluxes.values()]
        self.internal_state_variables_size = [
            max(1, v) for v in self._behaviour.internal_state_variables.values()
        ]
        self.gradients = np.zeros((ngauss, sum(self.gradients_size)))
        self.fluxes = np.zeros((ngauss, sum(self.fluxes_size)))
        self.internal_state_variables = np.zeros(
            (ngauss, sum(self.internal_state_variables_size))
        )
        self.internal_state_variables_pos = np.concatenate(
            ([0], self.internal_state_variables_size)
        ).cumsum()

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
        start = self.internal_state_variables_pos[index]
        size = self.internal_state_variables_size[index]
        pos = np.arange(start, start + size)
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
