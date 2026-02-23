from basix import index
import numpy as np
from scipy.spatial import cKDTree
import mui4py
from dolfinx import mesh, fem
import ufl

DOMAIN = "HeatSolverPy"

class Boundary():

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK):
        self.name = name
        self.position = position
        self.axis = axis
        self.MARK = MARK

        self.facets = mesh.locate_entities_boundary(domain, fdim, self.on_boundary)
        self.facet_tag = mesh.meshtags(domain, fdim, self.facets, np.full_like(self.facets, MARK))

        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tag)

        self.dofs = fem.locate_dofs_topological(V, fdim, self.facets)

        # Extract coordinates of DOFs
        self.dof_coords = V.tabulate_dof_coordinates().reshape((-1, gdim))[self.dofs]
        self.tree = cKDTree(self.dof_coords)
        self.values = np.zeros(self.dof_coords.shape[0])

        #Boundary function
        self.u_bc = fem.Function(V)
        self.u_bc.name = "u_bc"
        self.u_bc.interpolate(self)
        # Assume dirichlet boundary
        self.bc = fem.dirichletbc(self.u_bc, self.dofs)

        self.measure = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tag)
        

    def name(self):
        return self.name
    
    def coordinates(self):
        return self.dof_coords
    
    def on_boundary(self, x):
        return np.isclose(x[self.axis], self.position)
    
    def set_bc_val(self, value):
        self.values[:] = value
    
    def interpolate(self):
        self.u_bc.interpolate(self)
    
    def __call__(self, x, tol=1e-12):
        values = np.zeros(x.shape[1])

        dofs = x.T
        for index in range(len(dofs)):
            distance_to_neighbour, neighbour_index = self.tree.query(dofs[index])

            if distance_to_neighbour < tol:
                values[index] = self.values[neighbour_index]
        
        return values

class DirichletBoundary(Boundary):
    
    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, value):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK)
        self.values[:] = value
    

class CoupledBoundary(Boundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK)
        self.MPI_COMM_WORLD = MPI_COMM_WORLD
        self.PYTHON_COMM_WORLD = PYTHON_COMM_WORLD
        
        # Interface setup
        self.solverNum = self.MPI_COMM_WORLD.Get_rank()
        self.numSolvers = self.MPI_COMM_WORLD.Get_size()

         # boundary is a plane
        dims = 2

        # define config
        config = mui4py.Config(dims, mui4py.FLOAT64)

         # create interface
        interfaces = [name]
        self.interface = mui4py.create_unifaces(DOMAIN, interfaces, config)[name]

        self.interface.set_data_types({"temp": mui4py.FLOAT64,
                                       "flux": mui4py.FLOAT64,
                                       "coupling": mui4py.FLOAT64,
                                       "weight": mui4py.FLOAT64})
        

        # define samplers
        self.s_sampler = mui4py.SamplerPseudoNearestNeighbor(0.2)
        self.t_sampler_exact = mui4py.TemporalSamplerExact()
        self.t_sampler = mui4py.TemporalSamplerLinear()


class NeumannCoupledBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, q_flux):
         # define neumann boundary condition, v=TestFunction(V)
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD)
        self.q_flux = q_flux
        self.flux_values = np.zeros(self.dof_coords.shape[0])
       

    def update(self, T, iteration):

 
        #push temperature to neighbour
        self.interface.push_many("temp", self.dof_coords[:, 1:], T)
        self.interface.commit( iteration )
        # fetch heat flux from neighbour
        self.flux_values = self.interface.fetch_many("flux", self.dof_coords[:, 1:], iteration, self.s_sampler, self.t_sampler_exact)
        for i, dof in enumerate(self.dofs):
            self.q_flux.x.array[dof] = self.flux_values[i]
        self.q_flux.x.scatter_forward()

   
class DirichletCoupledBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD)

    def update(self, Q, iteration):
        # push heat_flux to neighbour
        self.interface.push_many("flux", self.dof_coords[:, 1:], Q)
        self.interface.commit( iteration )

        # fetch temperature from neighbour
        self.values = self.interface.fetch_many("temp", self.dof_coords[:, 1:], iteration, self.s_sampler, self.t_sampler)
        self.interface.forget(iteration)


class LinearInterpolationBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, kDelta: float):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD)

        self.kDelta = kDelta

    def update(self, T1, iteration):
        self.interface.push_many("temp", self.dof_coords, T1)
        self.interface.commit(iteration)

        nbrKDelta = self.interface.fetch_many("weight", self.dof_coords, iteration, self.s_sampler, self.t_sampler)[0]
        boundaryWeight = nbrKDelta / (nbrKDelta + self.kDelta)

        nbrTemp = self.interface.fetch_many("temp", self.dof_coords, iteration, self.s_sampler, self.t_sampler)
        self.values = boundaryWeight * nbrTemp + (1 - boundaryWeight) * T1
        self.interface.forget(iteration)