import numpy as np
from dolfinx import mesh, fem, geometry
import ufl
from boundarys import NeumannCoupledBoundary, DirichletCoupledBoundary, DirichletBoundary, LinearInterpolationBoundary
from mpi4py import MPI


LEFT_MARK = 88  # marker for left boundary facets
RIGHT_MARK = 66  # marker for right boundary facets

#TODO: fix this, should come from input file
RIGHT_BOUNDARY_FIXED_VALUE = 273.15


class BoxMesh:
    def __init__(self, meshDimensions, meshLengths, COMM_WORLD):
        self.nx = meshDimensions[0] # mesh divisions in x-direction
        self.ny = meshDimensions[1] # mesh divisions in y-direction
        self.nz = meshDimensions[2] # mesh divisions in z-direction

        self.lx = meshLengths[0] # domain length in x-direction
        self.ly = meshLengths[1] # domain length in y-direction
        self.lz = meshLengths[2] # domain length in z-direction

        self.p_min = np.array([1.0, 0, 0], dtype=np.float64)  # domain min corner
        self.p_max = np.array([1.0+self.lx,  self.ly,  self.lz], dtype=np.float64)  # domain max corner
        # create mesh and identify boundary facets
        self.domain = mesh.create_box(COMM_WORLD, [self.p_min, self.p_max], [self.nx, self.ny, self.nz], 
                        cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)
        self.gdim = self.domain.geometry.dim
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.domain.topology.create_connectivity(self.fdim, self.tdim)

        # exterior facets indices (global entity indices)
        self.boundary_facets = mesh.exterior_facet_indices(self.domain.topology)

        # FacetNormal(domain) yields the outward normal on the domain boundary facets
        self.n = ufl.FacetNormal(self.domain)
        #tree for computing collisions
        self.bb_tree = geometry.bb_tree(self.domain, self.tdim, padding=1e-10)
        
    
    def initialise_boundary_conditions(self, V, coef, kDelta, LOCAL_COMM_WORLD, MPI_COMM_WORLD, q_flux):
        bcs = []
        self.coupled_boundary_type = self.findBoundaryType(coef, LOCAL_COMM_WORLD, MPI_COMM_WORLD)

        print(self.coupled_boundary_type)
        
        if self.coupled_boundary_type == "dirichlet":
            self.left_boundary = DirichletCoupledBoundary("CoupledBoundary", self.domain, V,
                                                       self.p_min[0], 0, self.fdim, self.gdim, LEFT_MARK, LOCAL_COMM_WORLD, MPI_COMM_WORLD)
            bcs.append(self.left_boundary.bc)
        else:
            self.left_boundary = NeumannCoupledBoundary("CoupledBoundary", self.domain, V, self.p_min[0], 0, self.fdim, self.gdim, LEFT_MARK, LOCAL_COMM_WORLD, MPI_COMM_WORLD, q_flux)

        self.right_boundary = DirichletBoundary("DirichletBoundary", self.domain, V, self.p_max[0], 0, self.fdim, self.gdim, RIGHT_MARK, 274.15)
        bcs.append(self.right_boundary.bc)

        return bcs

    def findBoundaryType(self, coef, LOCAL_COMM_WORLD, MPI_COMM_WORLD):
        sendbuf = np.zeros(1)
        sendbuf[0] = coef
        recvbuf = np.zeros(1)

        pythonRank = LOCAL_COMM_WORLD.Get_rank()

        if pythonRank == 0:
            assert(MPI_COMM_WORLD.Get_rank() == 1)
            MPI_COMM_WORLD.Sendrecv([sendbuf, MPI.DOUBLE], dest=0, recvbuf=[recvbuf, MPI.DOUBLE], source=0)

            coefNeighbour = recvbuf[0]
        else:
            coefNeighbour = None
        # TODO BROADCAST coefNeighbour

        print(coefNeighbour)

        if coef/coefNeighbour <= 1:
            return "dirichlet"
        else:
            return "neumann"
    
    def update_boundary_conditions(self, val, iteration):
        self.left_boundary.update(val, iteration)
        self.left_boundary.interpolate()
        self.right_boundary.interpolate()

    def find_field_values_at(self, points, u_func):
        """"To evaluate specific points on field we need to find the cells that collide with these points,
        Note this method does not consider distributed meshes, will be deprecated soon anyway"""
        cells = []
        cell_candidates = geometry.compute_collisions_points(self.bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points)
        for i, point in enumerate(points):
            links = colliding_cells.links(i)
            if len(links) > 0:
                cells.append(links[0])
        return u_func.eval(points, cells).reshape(-1)
