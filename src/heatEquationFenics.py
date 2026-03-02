
import mpi4py.MPI as MPI
 # Necessary to avoid hangs at PETSc vector communication
from heatEquation import HeatEquation
import ufl
from petsc4py import PETSc

from meshGeneration import BoxMesh
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, set_bc, apply_lifting
from boundarys import NeumannCoupledBoundary, DirichletCoupledBoundary, DirichletBoundary, LinearInterpolationBoundary
from dolfinx import fem


LEFT_MARK = 88  # marker for left boundary facets
RIGHT_MARK = 66  # marker for right boundary facets

class HeatEquationFenics(HeatEquation):
    def __init__(self, LOCAL_COMM_WORLD, GLOBAL_COMM_WORLD):
        super().__init__()

        self.LOCAL_COMM_WORLD = LOCAL_COMM_WORLD
        self.GLOBAL_COMM_WORLD = GLOBAL_COMM_WORLD

        # set case parameters from input file
        meshLengths = (self.case_parameters['lx'], self.case_parameters['ly'], self.case_parameters['lz'])
        thermal_conductivity = self.case_parameters['kappa']
        thermal_diffusivity = self.case_parameters['alpha']

        # set solver parameters from input file 
        meshDimensions = (self.solver_parameters['nx'], self.solver_parameters['ny'], self.solver_parameters['nz'])
        poly_order = self.solver_parameters['poly_order']
        self.dt = self.solver_parameters['deltaT']
        self.end_time = self.solver_parameters['end_time']
        self.inner_loop_iterations = self.solver_parameters['inner_loop_iterations']
        self.writeInterval = self.solver_parameters['write_interval']
        self.num_steps = int(self.end_time/self.dt)

        # Create boxMesh
        self.mesh = BoxMesh(meshDimensions, meshLengths, self.LOCAL_COMM_WORLD)

        # Define function spaces
        self.V = fem.functionspace(self.mesh.domain, ("Lagrange", poly_order))
        self.V_out = fem.functionspace(self.mesh.domain, ("Lagrange", 1))
        
        # Define Constants
        self.thermal_conductivity = fem.Constant(self.mesh.domain, PETSc.ScalarType(thermal_conductivity))
        self.thermal_diffusivity = fem.Constant(self.mesh.domain, PETSc.ScalarType(thermal_diffusivity))
        f = fem.Constant(self.mesh.domain, PETSc.ScalarType(0.0))  # source term

        # Functions
        self.u_n = fem.Function(self.V)
        self.u_n.name = "u_n"

        self.q_flux = fem.Function(self.V)
        self.q_flux.name = "q_flux"

        self.uh = fem.Function(self.V)
        self.uh.name = "uh"

        self.uh_out = fem.Function(self.V_out)
        self.uh_out.interpolate(self.uh)

        # Trial and Test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Initial Conditions
        self.initialise_temperature_field(self.case_parameters['initial_temp'])
        
        # Boundary Conditions
        self.bcs = []
        self.right_boundary = DirichletBoundary("DirichletBoundary", self.mesh.domain, self.V, self.mesh.p_max[0], 0, 
                                                self.mesh.fdim, self.mesh.gdim, RIGHT_MARK, self.case_parameters['right_bc_temp'])
        
        self.bcs.append(self.right_boundary.bc)
        if self.solver_parameters['coupled_boundary_type'] == "neumann":
            self.left_boundary = NeumannCoupledBoundary("CoupledBoundary", self.mesh.domain, self.V, self.mesh.p_min[0], 0, 
                                                self.mesh.fdim, self.mesh.gdim, LEFT_MARK, self.LOCAL_COMM_WORLD, self.GLOBAL_COMM_WORLD, self.uh, self.q_flux)

        elif self.solver_parameters['coupled_boundary_type'] == "dirichlet":
            self.left_boundary = DirichletCoupledBoundary("CoupledBoundary", self.mesh.domain, self.V, self.mesh.p_min[0], 0, 
                                                self.mesh.fdim, self.mesh.gdim, LEFT_MARK, self.LOCAL_COMM_WORLD, self.GLOBAL_COMM_WORLD, self.uh, poly_order)
            self.left_boundary.set_bc_val(self.case_parameters['left_bc_temp'])
            self.bcs.append(self.left_boundary.bc)
        elif self.solver_parameters['coupled_boundary_type'] == "linearInterpolation":
            kDelta = self.thermal_conductivity.value/self.mesh.dx
            self.left_boundary = LinearInterpolationBoundary("CoupledBoundary", self.mesh.domain, self.V, self.mesh.p_min[0], 0, 
                                                self.mesh.fdim, self.mesh.gdim, LEFT_MARK, self.LOCAL_COMM_WORLD, self.GLOBAL_COMM_WORLD, kDelta)
            self.left_boundary.set_bc_val(self.case_parameters['left_bc_temp'])
            self.bcs.append(self.left_boundary.bc)
        else:
            raise Exception("Invalid coupled boundary type , "  + self.solver_parameters['coupled_boundary_type'] + ", specified")


        # heat equation in FEM form
        F = (u - self.u_n) / self.dt * v * ufl.dx + self.thermal_diffusivity * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - (self.q_flux/self.thermal_conductivity) * v * self.left_boundary.ds

        self.a = fem.form(ufl.lhs(F))
        self.L = fem.form(ufl.rhs(F))

        # Assemble matrix with BCs applied
        A = assemble_matrix(self.a, bcs=self.bcs)
        A.assemble()
        self.b = create_vector(self.L)

        # PETSc KSP solver setup
        self.solver = PETSc.KSP().create(self.LOCAL_COMM_WORLD)
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)


    def initialise_temperature_field(self, T0):
        self.uh.x.array[:] = T0  # initial condition: uniform temperature
        self.u_n.x.array[:] = T0

    def update_boundary_conditions(self, iteration):
        self.left_boundary.update(iteration)
        self.left_boundary.interpolate()
        self.right_boundary.interpolate()

    def solve(self):
        # Solve the heat equation using the current boundary conditions
         # Assemble RHS with current u_n
        with self.b.localForm() as loc_b:
            loc_b.set(0.0)
        assemble_vector(self.b, self.L)

        # Apply BCs to RHS
        apply_lifting(self.b, [self.a], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs)

        # Solve linear system A * uh = b
        self.solver.solve(self.b, self.uh.x.petsc_vec)
        self.uh.x.scatter_forward()
        
    def update_time(self):
        # Update u_n for next time step
        self.u_n.x.array[:] = self.uh.x.array

        self.uh_out.interpolate(self.uh)