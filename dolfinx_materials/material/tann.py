"""
Created on May 24, 2022
@author: Ioannis Stefanou & Filippo Masi
"""
from dolfinx_materials.material import Material
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO and WARNING messages are not printed
# = '0' all messages are logged (default behavior)
# = '1' INFO messages are not printed
# = '3' INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf

tf.keras.backend.set_floatx("float64")  # set tensorflow floating precision

import numpy as np  # manipulation of arrays
from dolfinx.common import Timer


class TannMaterial(Material):
    """
    AI Material class
    """

    def __init__(self, ANN_filename, nb_isv):
        """
        Load ANN network

        :param ANN_filename: ANN filename with path
        :type string
        :param nb_isv: number of internal state variables
        :type integer
        """
        super().__init__()
        self.model = tf.saved_model.load(ANN_filename)
        self.nb_isv = nb_isv
        self.dt = 0.0

    @property
    def internal_state_variables(self):
        return {
            "ivars": self.nb_isv,
            "free_energy": 1,
            "dissipation": 1,
        }

    def predict_AI_wrapper(self, deGP, svarsGP_t):
        """
        User material at a Gauss point

        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array

        :return: generalized stress at GP output, state variables at GP - output, jacobian at GP - output
        :rtype: numpy array, numpy array, numpy array


        Wrapper to predict material response at the Gauss point via an Artificial Neural Network (model)

        The model is called with inputs of size
        :param inputs: state variables, generalized deformation vector at GP
        :type inputs: numpy array (concatenate)
        :shape inputs: (1, 14 + self.nb_isv) = (1, 36), axis=0 represents the batch size (and should not modified)
        :call self.model(inputs,training=False)
        :return call: stressGP_t, svarsGP_t, dsdeGP_t with batch_size = 1, thus [0] squeeze the arrays along axis=0

        Note: the material response is normalized.
        """
        with Timer("TANN_predict: expand"):
            inputs = np.concatenate((svarsGP_t[:, : 12 + self.nb_isv], deGP), axis=1)
        with Timer("TANN_predict: infer"):
            stressGP_t, svarsGP_t, dsdeGP_t = self.model(inputs, training=False)
        with Timer("TANN_predict: retype"):
            stressGP_t = stressGP_t.numpy()
            svarsGP_t = svarsGP_t.numpy()
            dsdeGP_t = dsdeGP_t.numpy()
        return stressGP_t, svarsGP_t, dsdeGP_t

    def usermatGP(
        self, stressGP_t, deGP, svarsGP_t, dsdeGP_t, dt, GP_id, aux_deGP=np.zeros(1)
    ):
        """
        User material at a Gauss point

        :param stressGP_t: generalized stress at GP - input/output
        :type stressGP_t: numpy array
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param aux_deGP: auxiliary generalized deformation vector at GP - input
        :type aux_deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        :param dsdeGP_t: jacobian at GP - output
        :type dsde_t: numpy array
        :param dt: time increment
        :type dt: double
        :param GP_id: Gauss Point id (global numbering of all Gauss Points in the problem) - for normal materials is of no use
        :type GP_id: integer
        """
        stressGP_t[:], svarsGP_t[:], dsdeGP_t[:] = self.predict_AI_wrapper(
            deGP, svarsGP_t
        )

        return

    def constitutive_update_vectorized(self, eps, state):
        Ct = np.zeros((eps.shape[0], 36))
        deps = eps - state["Strain"]
        sig = state["Stress"]
        with Timer("TANN: concatenate"):
            state_vars = np.concatenate(
                (
                    state["Strain"],
                    state["Stress"],
                    state["ivars"],
                    state["free_energy"],
                    state["dissipation"],
                ),
                axis=1,
            )
        with Timer("TANN: inference"):
            self.usermatGP(
                sig,
                deps,
                state_vars,
                Ct,
                self.dt,
                0,
            )
        with Timer("TANN: return"):
            state["Strain"] = eps
            state["Stress"] = sig
            state["ivars"] = state_vars[:, 12 : 12 + self.nb_isv]
            state["free_energy"] = state_vars[:, [-2]]
            state["dissipation"] = state_vars[:, [-1]]

        return sig, Ct.reshape((-1, 6, 6))
