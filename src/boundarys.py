from basix import index
import numpy as np
from scipy.spatial import cKDTree
import mui4py
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl

DOMAIN = "HeatSolverFenics"

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

        # Normal vector
        self.n = ufl.FacetNormal(domain)
        

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

    def update(self, iteration):
        return
    

class CoupledBoundary(Boundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK)
        self.MPI_COMM_WORLD = MPI_COMM_WORLD
        self.PYTHON_COMM_WORLD = PYTHON_COMM_WORLD
        
        # Interface setup
        self.rank = self.MPI_COMM_WORLD.Get_rank()
        self.python_rank = self.PYTHON_COMM_WORLD.Get_rank()

        print("HELLO")
        # Create a communicator of only ranks at the interface
        interface_ranks = MPI_COMM_WORLD.Split(color=1, key=self.rank)

        print("HELLO")

        print("Rank {:d} has {:d} facets".format(self.rank, self.facets.shape[0]))

        dims = 3

        # define config
        config = mui4py.Config(dims, mui4py.FLOAT64)

         # create interface
        interfaces = [name]
        self.interface = mui4py.create_unifaces(DOMAIN, interfaces, config, world = interface_ranks)[name]

        self.interface.set_data_types({"temp": mui4py.FLOAT64,
                                       "flux": mui4py.FLOAT64,
                                       "coupling": mui4py.FLOAT64,
                                       "weight": mui4py.FLOAT64})
        
        # coordinates must be rounded to account for differing precisions betweeen openfoam and python
        self.coupling_coordinates = np.round(self.dof_coords, decimals=8)
        
        # define samplers
        self.s_sampler = mui4py.SamplerPseudoNearestNeighbor(0.2)
        self.t_sampler_exact = mui4py.TemporalSamplerExact()
        self.t_sampler = mui4py.TemporalSamplerLinear()

        # define aitken relaxation alg
        self.a_algorithm = mui4py.AlgorithmAitken(0.5, 1.0, config=config)

        # define temperature field reference
        self.u_func = u_func


class NeumannCoupledBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func, q_flux):
         # define neumann boundary condition, v=TestFunction(V)
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func)
        self.q_flux = q_flux
        self.flux_values = np.zeros(self.dof_coords.shape[0])


    def update(self, iteration):
        # get temperature at the boundary
        T = self.u_func.x.array[self.dofs]
   
        #push temperature to neighbour
        self.interface.push_many("temp", self.coupling_coordinates, T)
        self.interface.commit( iteration )

        # fetch heat flux from neighbour
        self.flux_values = self.interface.fetch_many("flux", self.coupling_coordinates, iteration, self.s_sampler, self.t_sampler_exact)

        self.q_flux.x.array[self.dofs] = -self.flux_values
        self.q_flux.x.scatter_forward()

   
class DirichletCoupledBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func, kappa, poly_order):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func)

        # Create vector function space for flux
        self.V_g = fem.functionspace(domain, ("Lagrange", poly_order, (gdim, )))


        # Create a function for flux evaluation
        self.flux = fem.Function(self.V_g)
        self.flux.name = "flux"

        # Heat flux expression
        self.flux_expr = fem.Expression(-kappa * ufl.grad(self.u_func), self.V_g.element.interpolation_points())
        

    def update(self, iteration):
        Q = self.compute_heat_flux()
        # push heat_flux to neighbour
        self.interface.push_many("flux", self.coupling_coordinates, Q)
        self.interface.commit( iteration )

        # fetch temperature from neighbour
        self.values = self.interface.fetch_many("temp", self.coupling_coordinates, iteration, self.s_sampler, self.t_sampler)
        self.interface.forget(iteration)

    def compute_heat_flux(self):
        self.flux.interpolate(self.flux_expr)

        flux_vals = self.flux.x.array.reshape((-1, self.dof_coords.shape[1]))[self.dofs][:, 0]

        print(flux_vals)
        # Evaluate flux at these dofs
        return -flux_vals

# Deprecated for now
class LinearInterpolationBoundary(CoupledBoundary):

    def __init__(self, name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func, dx, kappa):
        super().__init__(name, domain, V, position, axis, fdim, gdim, MARK, PYTHON_COMM_WORLD, MPI_COMM_WORLD, u_func)

        self.kDelta = kappa/dx

        self.domain = domain

        #tree for computing collisions
        self.bb_tree = geometry.bb_tree(domain, fdim+1, padding=1e-10)
        
        self.internalPoints = self.dof_coords + np.array([dx, 0, 0])

    def update(self, iteration):

        T1 = self.getInternalTemp()

        self.interface.push("weight", self.coupling_coordinates[0], self.kDelta)

        self.interface.push_many("temp", self.coupling_coordinates, T1)
        self.interface.commit(iteration)

        nbrKDelta = self.interface.fetch("weight", self.coupling_coordinates[0], iteration, self.s_sampler, self.t_sampler)
        boundaryWeight = nbrKDelta / (nbrKDelta + self.kDelta)

        nbrTemp = self.interface.fetch_many("temp", self.coupling_coordinates, iteration, self.s_sampler, self.t_sampler)

        self.values = boundaryWeight * nbrTemp + (1 - boundaryWeight) * T1
        self.interface.forget(iteration)

    def getInternalTemp(self):
        """"To evaluate specific points on field we need to find the cells that collide with these points,
        Note this method does not consider distributed meshes"""
        cells = []
        cell_candidates = geometry.compute_collisions_points(self.bb_tree, self.internalPoints)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, self.internalPoints)
        for i, point in enumerate(self.internalPoints):
            links = colliding_cells.links(i)
            if len(links) > 0:
                cells.append(links[0])
        return self.u_func.eval(self.internalPoints, cells).reshape(-1)
