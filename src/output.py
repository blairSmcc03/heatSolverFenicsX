import numpy as np
from dolfinx import io

class Output:
    def __init__(self, domain, T, LOCAL_COMM_WORLD):
        self.output_directory = "output/"
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


    def plotTemperature(self):
        self.pcm.set_array(self.T.T)
        self.axis.set_title("Temperature at time t: {:.3f}s".format(self.time))
        plt = None

        fig, ax = plt.subplots()

        xRange = np.linspace(self.solverNum*self.width, (self.solverNum+1)*self.width, self.nodes)
        tempData = np.average(self.T, axis=1)

        slope, intercept = np.polyfit(xRange, tempData, 1)

        ax.plot(xRange, tempData)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Temp (K)")
        ax.set_title("Temp from Solver {:d}. Equation of line: T = {:.3f}x + {:.2f}".format(self.solverNum, slope, intercept))

        
        print("Solver {:d}       Left boundary has temperature: {:3f}K         Right boundary has temperature: {:3f}K".format(self.solverNum, np.average(self.T[0]), np.average(self.T[-1])))

        x_axis = (self.T[:, int(self.nodes/2)]+self.T[:, int((self.nodes/2)+1)])/2
        f = open("pythonHeatData.xy", "w")
        x = 1.0
        for i in range(self.nodes):
            s = str(x) + "  " + str(x_axis[i]) + "\n"
            x += self.dx

            f.write(s)

        f.close()

        plt.show()

        print("Solver: {:d}, Number of Solvers: {:d}".format(self.solverNum, self.numSolvers))

        print("Created heat solver {:d} with size {:f}m x {:f}m, {:d} nodes, diffusivity {:10f} and kappa {:f}".format(self.solverNum, self.height, self.width, self.nodes**2, self.alpha, self.kappa))

        print("Using coupling method " + self.couplingMethod)