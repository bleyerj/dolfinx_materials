from dolfinx_materials.material import Material
import jax

from functools import wraps, partial

jax.config.update("jax_enable_x64", True)  # use double-precision


def tangent_AD(consitutive_upate_method):
    @wraps(consitutive_upate_method)
    def wrapper(self, *args):
        constitutive_update_fun = partial(consitutive_upate_method, self)
        return jax.jacfwd(constitutive_update_fun, argnums=0, has_aux=True)(*args)

    return wrapper


class JAXMaterial(Material):
    """This class is used to define a material behavior implemented with JAX."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batched_constitutive_update = jax.jit(
            jax.vmap(self.constitutive_update, in_axes=(0, 0, None))
        )
