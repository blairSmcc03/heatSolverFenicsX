from time import time
import numpy as np
from dolfinx import io

class Output:
    def __init__(self, domain, T, LOCAL_COMM_WORLD):
        self.output_directory = "solid/output/"
        self.xdmf_filename = self.output_directory + "fenicsx_solid_data.xdmf"  # XDMF output filename

        self.LOCAL_COMM_WORLD = LOCAL_COMM_WORLD

        self.xdmf = io.XDMFFile(self.LOCAL_COMM_WORLD, self.xdmf_filename, "w")
        
        if self.LOCAL_COMM_WORLD.rank == 0:
            """Write FEniCSx mesh object to XDMF file"""
            self.xdmf.write_mesh(domain)


    
    def writeFunction(self, function, time):
        """Write field data at specific time to xdmf file."""
        self.xdmf.write_function(function, time)
        print("FEniCSx time: {:f}".format(time))
    
    def close(self):
        self.xdmf.close()
        self.residual_file.close()
