import jax
import jaxmat.materials as jm
import numpy as np
from jaxmat.tensors import SymmetricTensor2

from dolfinx_materials.jaxmat import JAXMaterial

jax.config.update("jax_enable_x64", True)


def test_state_is_restored_between_increments():
    """Load to 2 % then unload: the update must start from the previous converged state, so the
    cumulated plastic strain stays constant on unloading and matches jaxmat called directly."""
    behavior = jm.vonMisesIsotropicHardening(
        elasticity=jm.LinearElasticIsotropic(E=70e3, nu=0.3),
        yield_stress=jm.VoceHardening(sig0=350.0, sigu=500.0, b=1e3),
    )
    material = JAXMaterial(behavior)
    material.set_data_manager(1)
    state = jax.tree.map(lambda x: x[0], behavior.init_state(1))
    for eyy in np.r_[np.linspace(0.002, 0.02, 10), np.linspace(0.018, 0.01, 5)]:
        eps = np.array([-0.5 * eyy, eyy, 0.0, 0.0, 0.0, 0.0])
        stress, isv, _ = material.integrate(eps[None])
        material.data_manager.update()
        sig, state = behavior.constitutive_update(SymmetricTensor2(array=eps), state, 0.0)
        np.testing.assert_allclose(np.asarray(isv)[0, 0], state.internal.p, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(np.asarray(stress)[0], sig.array, rtol=1e-8, atol=1e-8)
