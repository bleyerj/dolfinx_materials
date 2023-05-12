import numpy as np

class PythonMaterial:
    @property
    def name(self):
        return self.__class__.__name__
    
    def get_gradients(self):
        return {"eps": 6}
    def get_fluxes(self):
        return {"sig": 6}
    def get_internal_state_variables(self):
        return {}
    
    def get_variables(self):
        return {**self.get_gradients(), **self.get_fluxes(), **self.get_internal_state_variables()}
    
    def set_data_manager(self, ngauss):
        # Setting the material data manager
        self.data_manager = PythonDataManager(self, ngauss)
    
    def integrate(self, gradients):
        flux_array = []
        Ct_array = []
        for i, g in enumerate(gradients):
            state = self.data_manager.s0[i]
            flux, Ct = self.constitutive_update(g, state)
            flux_array.append(flux)
            Ct_array.append(Ct.ravel())
            self.data_manager.s1[i] = state
        return np.array(flux_array), np.array(Ct_array)
    
    def get_initial_state_dict(self):
        return self.data_manager.s0[:]
    
    def get_final_state_dict(self):
        return self.data_manager.s1[:]


class PythonDataManager:
    def __init__(self, behaviour, ngauss):
        num_gradients = sum([v for v in behaviour.get_gradients().values()])
        num_fluxes = sum([v for v in behaviour.get_fluxes().values()])
        self.K = np.zeros((num_fluxes, num_gradients))
        self.s0 = PythonMaterialStateManager(behaviour, ngauss)
        self.s1 = PythonMaterialStateManager(behaviour, ngauss)
    
    def update(self):
        self.s0.update(self.s1)
        
    def revert(self):
        self.s1.update(self.s0)
        
    
class PythonMaterialStateManager:
    def __init__(self, behaviour, ngauss):
        self._behaviour = behaviour
        self.n = ngauss
        self.gradients_stride = [v for v in self._behaviour.get_gradients().values()]
        self.fluxes_stride = [v for v in self._behaviour.get_fluxes().values()]
        self.internal_state_variables_stride = [v for v in self._behaviour.get_internal_state_variables().values()]
        self.gradients = np.zeros((ngauss, sum(self.gradients_stride)))
        self.fluxes = np.zeros((ngauss, sum(self.fluxes_stride)))
        self.internal_state_variables = np.zeros((ngauss, sum(self.internal_state_variables_stride)))
    
    def update(self, other):
        self.gradients = other.gradients
        self.fluxes = other.fluxes
        self.internal_state_variables = other.internal_state_variables

    def get_flux_index(self, name):
        index = list(self._behaviour.get_fluxes().keys()).index(name)
        pos = range(index, index+self._behaviour.get_fluxes()[name])
        return pos
    

    def get_gradient_index(self, name):
        index = list(self._behaviour.get_gradients().keys()).index(name)
        pos = range(index, index+self._behaviour.get_gradients()[name])
        return pos
    

    def get_internal_state_variable_index(self, name):
        index = list(self._behaviour.get_internal_state_variables().keys()).index(name)
        pos = range(index, index+self._behaviour.get_internal_state_variables()[name])
        return pos
    
    def __getitem__(self, i):
        state = {}
        for key, value in self._behaviour.get_gradients().items():
            pos = self.get_gradient_index(key)
            state.update({key: self.gradients[i, pos]})
        for key, value in self._behaviour.get_fluxes().items():
            pos = self.get_flux_index(key)
            state.update({key: self.fluxes[i, pos]})
        for key, value in self._behaviour.get_internal_state_variables().items():
            pos = self.get_internal_state_variable_index(key)
            state.update({key: self.internal_state_variables[i, pos] })
        return state

    def __setitem__(self, i, state):
        state_copy = state.copy()
        for key, value in self._behaviour.get_gradients().items():
            pos = self.get_gradient_index(key)
            self.gradients[i, pos] = state[key]
            state_copy.pop(key)
        for key, value in self._behaviour.get_fluxes().items():
            pos = self.get_flux_index(key)
            self.fluxes[i, pos] = state[key]
            state_copy.pop(key)
        for key, value in self._behaviour.get_internal_state_variables().items():
            pos = self.get_internal_state_variable_index(key)
            self.internal_state_variables[i, pos] = state[key]
            state_copy.pop(key)
        assert len(state_copy)==0, "Material state contains unknown field to update with."