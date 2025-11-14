#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Interface to jaxmat behaviors.

@Author  :   Jeremy Bleyer, Ecole Nationale des Ponts et Chaussées, Navier
@Contact :   jeremy.bleyer@enpc.fr
@Time    :   01/11/2025
"""
from dolfinx.common import Timer
from dolfinx_materials.material import Material
import jax
import equinox as eqx
import numpy as np
from jaxmat.tensors import Tensor
from jaxmat.materials import SmallStrainBehavior, FiniteStrainBehavior


def get_shape(val):
    if isinstance(val, Tensor):
        return val.array_shape[0]
    else:
        shape = val.shape
        if shape == ():
            return 1
        else:
            return shape[0]


class DataManager:
    def __init__(self, material, ngauss):
        num_gradients = sum([v for v in material.gradients.values()])
        num_fluxes = sum([v for v in material.fluxes.values()])
        self.K = np.zeros((num_fluxes, num_gradients))
        self.jaxmat_state = material.behavior.init_state(ngauss)
        self.s0 = {}
        self.s1 = {}

    def update(self):
        self.s0 = self.s1

    def revert(self):
        self.s1 = self.s0


def hcat_mixed(arrays):
    """Horizontally concatenate arrays of shape (d,) or (d, n)."""
    prepped = []
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[:, None]  # make it (d, 1)
        elif a.ndim != 2:
            raise ValueError(
                f"Unexpected array shape {a.shape}, expected (d,) or (d,n)."
            )
        prepped.append(a)
    return np.concatenate(prepped, axis=1)


def merge_special_key(d: dict, special_key: str) -> dict:
    """Flatten the contents of `special_key` into the top-level dict."""
    out = {}
    for k, v in d.items():
        if k == special_key:
            out.update(v.__dict__)  # merge its subkeys
        else:
            out[k] = v  # keep normal keys
    return out


@eqx.filter_jit
def jaxmat_to_dolfinx_state(jaxmat_state):
    def flatten(x):
        if isinstance(x, Tensor):
            return x.array
        else:
            return x

    # flatten Tensor objects into array repr
    dolfinx_state = jax.tree.map(
        flatten, jaxmat_state, is_leaf=lambda x: isinstance(x, Tensor)
    )
    #  merge `internal` sub dict with other entries
    dolfinx_state = merge_special_key(dolfinx_state.__dict__, "internal")
    return dolfinx_state


def replace_elem(key_type, val):
    if issubclass(key_type, Tensor):
        return key_type(array=val)
    else:
        return val


def _dolfinx_to_jaxmat_state(dolfinx_state, jaxmat_state):
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


def eqx_to_flatdict(module):
    out = {}

    def _walk(x, prefix=""):
        if isinstance(x, eqx.Module):
            for name, val in vars(x).items():
                _walk(val, f"{prefix}.{name}" if prefix else name)
        else:
            out[prefix] = x

    _walk(module)
    return out


@eqx.filter_jit
def dolfinx_to_jaxmat_state(dx_state, jx_state):
    eqx.filter_vmap(_dolfinx_to_jaxmat_state, in_axes=0)(dx_state, jx_state)
    return jx_state


class JAXMaterial(Material):
    """Converts a `jaxmat` behavior into a dolfinx-compatible behavior."""

    def __init__(self, behavior, jit=True):
        self.behavior = behavior
        self.material_properties = eqx_to_flatdict(self.behavior)
        self.batched_constitutive_update = eqx.filter_vmap(
            jax.jacfwd(self.constitutive_update, argnums=0, has_aux=True),
            in_axes=(0, 0, None),
            out_axes=(0, 0),
        )
        if jit:
            self.batched_constitutive_update = eqx.filter_jit(
                self.batched_constitutive_update
            )
        self._first_pass = True

    def constitutive_update(self, gradients, state, dt):
        # transform flattened gradient into correct jaxmat Tensor
        tensor_grad = getattr(
            self.data_manager.jaxmat_state, self.gradient_names[0]
        ).__class__(array=gradients)
        stress, new_state = self.behavior.constitutive_update(tensor_grad, state, dt)
        return stress.array, new_state

    @property
    def gradients(self):
        if isinstance(self.behavior, SmallStrainBehavior):
            return {"strain": 6}
        elif isinstance(self.behavior, FiniteStrainBehavior):
            return {"F": 9}
        else:
            raise NotImplementedError(
                "Only SmallStrainBehavior and FiniteStrainBehavior jaxmat behaviors are supported."
            )

    @property
    def fluxes(self):
        if isinstance(self.behavior, SmallStrainBehavior):
            return {"stress": 6}
        elif isinstance(self.behavior, FiniteStrainBehavior):
            return {"PK1": 9}
        else:
            raise NotImplementedError(
                "Only SmallStrainBehavior and FiniteStrainBehavior jaxmat behaviors are supported."
            )

    @property
    def internal_state_variables(self):
        return {
            key: get_shape(val)
            for key, val in self.behavior.init_state().internal.__dict__.items()
        }

    def set_data_manager(self, ngauss):
        # Setting the material data manager
        self.data_manager = DataManager(self, ngauss)

    def get_initial_state_dict(self):
        return self.data_manager.s0

    def get_final_state_dict(self):
        return self.data_manager.s1

    def set_initial_state_dict(self, state):
        self.data_manager.s0 = state

    def integrate(self, gradients, dt=0):
        with Timer("jaxmat: dolfinx to jaxmat conversion"):
            state = self.get_initial_state_dict()
            jx_state = self.data_manager.jaxmat_state
            jx_state = dolfinx_to_jaxmat_state(state, jx_state)

        if self._first_pass:
            timer_name = "jaxmat: First pass (includes jit compilation)"
            self._first_pass = False
        else:
            timer_name = "jaxmat: Constitutive update"
        with Timer(timer_name):
            Ct_array, new_jx_state = self.batched_constitutive_update(
                gradients, jx_state, dt
            )
        with Timer("jaxmat: jaxmat to dolfinx conversion"):
            new_dx_state = jaxmat_to_dolfinx_state(new_jx_state)
            self.data_manager.s1 = new_dx_state
            stress = new_dx_state.get(self.flux_names[0])
            isv = hcat_mixed(
                [new_dx_state[key] for key in self.internal_state_variable_names]
            )
        return (
            stress,
            isv,
            Ct_array,
        )
