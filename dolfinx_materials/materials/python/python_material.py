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
    
    # def integrate(self, gradients):
    #     for g in range(self.data_manager.ngauss):


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

    def get_internal_state_variable(self, name):
        index = list(self._behaviour.get_internal_state_variables().keys()).index(name)
        pos = range(index, index+self._behaviour.get_internal_state_variables()[name])
        return self.internal_state_variables[:, pos]