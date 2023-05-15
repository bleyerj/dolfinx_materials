#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFrontNonlinearMaterial class

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205)
@email: jeremy.bleyer@enpc.f
"""
import mgis.behaviour as mgis_bv

# from .gradient_flux import Var
# from .utils import compute_on_quadrature
# import dolfin
import subprocess
import os

mgis_hypothesis = {
    "plane_strain": mgis_bv.Hypothesis.PlaneStrain,
    "plane_stress": mgis_bv.Hypothesis.PlaneStress,
    "3d": mgis_bv.Hypothesis.Tridimensional,
    "axisymmetric": mgis_bv.Hypothesis.Axisymmetrical,
}


class MFrontMaterial:
    """
    This class handles the loading of a MFront behaviour through MGIS.
    """

    def __init__(
        self,
        path,
        name,
        hypothesis="3d",
        material_properties={},
        parameters={},
        rotation_matrix=None,
        dt=0,
    ):
        """
        Parameters
        -----------

        path : str
            path to the 'libMaterial.so' library containing MFront material laws
        name : str
            name of the MFront behaviour
        hypothesis : {"plane_strain", "3d", "axisymmetric"}
            modelling hypothesis
        material_properties : dict
            a dictionary of material properties. The dictionary keys must match
            the material property names declared in the MFront behaviour. Values
            can be constants or functions.
        parameters : dict
            a dictionary of parameters. The dictionary keys must match the parameter
            names declared in the MFront behaviour. Values must be constants.
        rotation_matrix : Numpy array, list of list, UFL matrix
            a 3D rotation matrix expressing the rotation from the global
            frame to the material frame. The matrix can be spatially variable
            (either UFL matrix or function of Tensor type)
        """
        self.path = path
        self.name = name
        # Defining the modelling hypothesis
        self.hypothesis = mgis_hypothesis[hypothesis]
        self.material_properties = material_properties
        self.rotation_matrix = rotation_matrix
        self.integration_type = (
            mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
        )
        self.dt = dt
        # Loading the behaviour
        try:
            self.load_behaviour()
        except:
            cwd = os.getcwd()
            install_path = "/".join(path.split("/")[:-2]) + "/"
            os.chdir(install_path)
            print(
                "Behaviour '{}' has not been found in '{}'.".format(
                    self.name, self.path
                )
            )
            print(
                "Attempting to compile '{}.mfront' in '{}'...".format(
                    self.name, install_path
                )
            )
            subprocess.run(
                ["mfront", "--obuild", "--interface=generic", self.name + ".mfront"]
            )
            os.chdir(cwd)
            self.load_behaviour()
        self.update_parameters(parameters)

    def load_behaviour(self):
        self.is_finite_strain = mgis_bv.isStandardFiniteStrainBehaviour(
            self.path, self.name
        )
        if self.is_finite_strain:
            # finite strain options
            bopts = mgis_bv.FiniteStrainBehaviourOptions()
            bopts.stress_measure = mgis_bv.FiniteStrainBehaviourOptionsStressMeasure.PK1
            bopts.tangent_operator = (
                mgis_bv.FiniteStrainBehaviourOptionsTangentOperator.DPK1_DF
            )
            self.behaviour = mgis_bv.load(bopts, self.path, self.name, self.hypothesis)
        else:
            self.behaviour = mgis_bv.load(self.path, self.name, self.hypothesis)

    def set_data_manager(self, ngauss):
        # Setting the material data manager
        self.data_manager = mgis_bv.MaterialDataManager(self.behaviour, ngauss)

    def update_parameters(self, parameters):
        for key, value in parameters.items():
            self.behaviour.setParameter(key, value)

    def update_material_property(self, name, values):
        for s in [self.data_manager.s0, self.data_manager.s1]:
            if type(values) in [int, float]:
                mgis_bv.setMaterialProperty(s, name, values)
            else:
                mgis_bv.setMaterialProperty(
                    s,
                    name,
                    values,
                    mgis_bv.MaterialStateManagerStorageMode.LocalStorage,
                )

    def update_external_state_variables(self, degree, mesh, external_state_variables):
        s = self.data_manager.s1
        for key, value in external_state_variables.items():
            if type(value) in [int, float]:
                mgis_bv.setExternalStateVariable(s, key, value)
            elif isinstance(value, dolfin.Constant):
                mgis_bv.setExternalStateVariable(s, key, float(value))
            else:
                if isinstance(value, Var):
                    value.update()
                    values = value.function.vector().get_local()
                else:
                    values = (
                        compute_on_quadrature(value, mesh, degree).vector().get_local()
                    )
                mgis_bv.setExternalStateVariable(
                    s, key, values, mgis_bv.MaterialStateManagerStorageMode.LocalStorage
                )

    def get_parameter(self, name):
        return self.behaviour.getParameterDefaultValue(name)

    def get_parameter_names(self):
        return self.behaviour.params

    def get_material_property_names(self):
        return [svar.name for svar in self.behaviour.mps]

    def get_external_state_variable_names(self):
        return [svar.name for svar in self.behaviour.external_state_variables]

    def get_internal_state_variable_names(self):
        return [svar.name for svar in self.behaviour.internal_state_variables]

    def get_gradient_names(self):
        return [svar.name for svar in self.behaviour.gradients]

    def get_flux_names(self):
        return [svar.name for svar in self.behaviour.thermodynamic_forces]

    def get_gradients(self):
        return {
            k: dim
            for k, dim in zip(self.get_gradient_names(), self.get_gradient_sizes())
        }

    def get_fluxes(self):
        return {k: dim for k, dim in zip(self.get_flux_names(), self.get_flux_sizes())}

    def get_internal_state_variables(self):
        return {
            k: dim
            for k, dim in zip(
                self.get_internal_state_variable_names(),
                self.get_internal_state_variable_sizes(),
            )
        }

    def get_variables(self):
        dict_grad = self.get_gradients()
        dict_flux = self.get_fluxes()
        dict_isv = self.get_internal_state_variables()
        return {**dict_grad, **dict_flux, **dict_isv}

    def get_material_property_sizes(self):
        return [
            mgis_bv.getVariableSize(svar, self.hypothesis)
            for svar in self.behaviour.mps
        ]

    def get_external_state_variable_sizes(self):
        return [
            mgis_bv.getVariableSize(svar, self.hypothesis)
            for svar in self.behaviour.external_state_variables
        ]

    def get_internal_state_variable_sizes(self):
        return [
            mgis_bv.getVariableSize(svar, self.hypothesis)
            for svar in self.behaviour.internal_state_variables
        ]

    def get_gradient_sizes(self):
        return [
            mgis_bv.getVariableSize(svar, self.hypothesis)
            for svar in self.behaviour.gradients
        ]

    def get_flux_sizes(self):
        return [
            mgis_bv.getVariableSize(svar, self.hypothesis)
            for svar in self.behaviour.thermodynamic_forces
        ]

    def get_tangent_block_names(self):
        return [(t[0].name, t[1].name) for t in self.behaviour.tangent_operator_blocks]

    def get_tangent_block_sizes(self):
        return [
            tuple([mgis_bv.getVariableSize(tt, self.hypothesis) for tt in t])
            for t in self.behaviour.tangent_operator_blocks
        ]

    def integrate(self, eps):
        for s in [self.data_manager.s0, self.data_manager.s1]:
            mgis_bv.setExternalStateVariable(s, "Temperature", 293.15)

        self.data_manager.s1.gradients[:, :] = eps
        mgis_bv.integrate(
            self.data_manager, self.integration_type, self.dt, 0, self.data_manager.n
        )

        _, n, m = self.data_manager.K.shape
        return self.data_manager.s1.thermodynamic_forces, self.data_manager.K.reshape(
            (-1, n * m)
        )

    def get_final_state_dict(self):
        state = {}
        buff = 0
        for i, s in enumerate(self.get_gradient_names()):
            block_shape = self.get_gradient_sizes()[i]
            state[s] = self.data_manager.s1.gradients[:, buff : buff + block_shape]
            buff += block_shape
        buff = 0
        for i, s in enumerate(self.get_flux_names()):
            block_shape = self.get_flux_sizes()[i]
            state[s] = self.data_manager.s1.thermodynamic_forces[
                :, buff : buff + block_shape
            ]
            buff += block_shape
        buff = 0
        for i, s in enumerate(self.get_internal_state_variable_names()):
            block_shape = self.get_internal_state_variable_sizes()[i]
            state[s] = self.data_manager.s1.internal_state_variables[
                :, buff : buff + block_shape
            ]
            buff += block_shape
        return state

    def rotate_gradients(self, gradient_vals, rotation_values):
        mgis_bv.rotateGradients(gradient_vals, self.behaviour, rotation_values)

    def rotate_fluxes(self, flux_vals, rotation_values):
        mgis_bv.rotateThermodynamicForces(flux_vals, self.behaviour, rotation_values)

    def rotate_tangent_operator(self, Ct_vals, rotation_values):
        mgis_bv.rotateTangentOperatorBlocks(Ct_vals, self.behaviour, rotation_values)
