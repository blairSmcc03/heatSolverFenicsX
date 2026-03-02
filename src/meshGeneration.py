import numpy as np
from dolfinx import mesh
import ufl

class BoxMesh:
    def __init__(self, meshDimensions, meshLengths, COMM_WORLD):
        self.nx = meshDimensions[0] # mesh divisions in x-direction
        self.ny = meshDimensions[1] # mesh divisions in y-direction
        self.nz = meshDimensions[2] # mesh divisions in z-direction

        self.lx = meshLengths[0] # domain length in x-direction
        self.ly = meshLengths[1] # domain length in y-direction
        self.lz = meshLengths[2] # domain length in z-direction

        self.dx = self.lx/self.nx
        self.dy = self.ly/self.ny
        self.dz = self.lz/self.nz

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
        
