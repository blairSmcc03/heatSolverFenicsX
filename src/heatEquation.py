from abc import ABC, abstractmethod
from input import Input

class HeatEquation(ABC):
    def __init__(self):
        
        input = Input()
        self.case_parameters = input.case_parameters
        self.solver_parameters = input.solver_parameters
            
    @abstractmethod
    def initialise_temperature_field(self, T0):
        pass

    @abstractmethod
    def solve(self):
        pass

  