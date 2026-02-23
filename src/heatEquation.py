from abc import ABC, abstractmethod

class HeatEquation(ABC):
    def __init__(self, meshDimensions, meshLengths, thermal_diffusivity, thermal_conductivity):
        self.nx, self.ny, self.nz = meshDimensions
        self.Lx, self.Ly, self.Lz = meshLengths

        # derived properties
        self.dx = self.Lx/self.nx
        self.dy = self.Ly/self.ny
        self.dz = self.Lz/self.nz

        self.thermal_diffusivity = thermal_diffusivity
        self.thermal_conductivity = thermal_conductivity


        self.cellArea = self.dy*self.dz
            
    @abstractmethod
    def initialise_temperature_field(self, T0):
        pass

    @abstractmethod
    def solve(self):
        pass

  