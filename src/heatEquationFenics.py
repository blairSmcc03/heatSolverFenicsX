
import mpi4py.MPI as MPI
 # Necessary to avoid hangs at PETSc vector communication
from heatEquation import HeatEquation
import ufl
from petsc4py import PETSc

from meshGeneration import BoxMesh
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, set_bc, apply_lifting, LinearProblem
import numpy as np
from dolfinx import fem
import math

class HeatEquationFenics(HeatEquation):
    def __init__(self, meshDimensions, meshLengths, thermal_diffusivity, thermal_conductivity, poly_order, LOCAL_COMM_WORLD, MPI_COMM_WORLD, dt=0.05):
        super().__init__(meshDimensions, meshLengths, thermal_diffusivity, thermal_conductivity)

        self.poly_order = poly_order
        self.LOCAL_COMM_WORLD = LOCAL_COMM_WORLD
        self.MPI_COMM_WORLD = MPI_COMM_WORLD
        self.dt = dt

        #self.iterationStep = self.findIterationNumber()
        # Mesh creation
        self.mesh = BoxMesh(meshDimensions, meshLengths, self.LOCAL_COMM_WORLD)
        # Define function spaces
        self.V = fem.functionspace(self.mesh.domain, ("Lagrange", poly_order))
        self.V_out = fem.functionspace(self.mesh.domain, ("Lagrange", 1))
        
        # Constants
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
        
        # Boundary Conditions
        # Values for coupling (see README.md for definitions)
        coef = self.dy*self.dz*self.thermal_conductivity.value/self.dx
        kDelta = self.thermal_conductivity.value/self.dx

        self.bcs = self.mesh.initialise_boundary_conditions(self.V, coef, kDelta, self.LOCAL_COMM_WORLD, self.MPI_COMM_WORLD, self.q_flux)

        # heat equation in FEM form
        F = (u - self.u_n) / dt * v * ufl.dx + self.thermal_diffusivity * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - (self.q_flux/self.thermal_conductivity) * v * self.mesh.left_boundary.ds

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

    def findIterationNumber(self):
        sendbuf = np.zeros(1)
        sendbuf[0] = self.dt
        recvbuf = np.zeros(1)

        pythonRank = self.LOCAL_COMM_WORLD.Get_rank()

        if pythonRank == 0:
            assert(self.MPI_COMM_WORLD.Get_rank() == 1)
            self.MPI_COMM_WORLD.Sendrecv([sendbuf, MPI.DOUBLE], dest=0, recvbuf=[recvbuf, MPI.DOUBLE], source=0)

            OFdt = recvbuf[0]
        else:
            OFdt = None

        # TODO BROADCAST OFdt
        iteration_step = math.ceil(self.dt/OFdt)
        return iteration_step


    def initialise_temperature_field(self, T0):
        self.uh.x.array[:] = T0  # initial condition: uniform temperature
        self.u_n.x.array[:] = T0

    def set_left_boundary_condition(self, T_L):
        self.mesh.left_boundary.set_bc_val(T_L)

    def set_right_boundary_condition(self, T_R):
        self.mesh.right_boundary.set_bc_val(T_R)

    def update_boundary_conditions(self, iteration):
        if self.mesh.coupled_boundary_type == "dirichlet":
            Q = self.compute_heat_flux(self.mesh.left_boundary)
            self.mesh.update_boundary_conditions(Q, iteration)
        else:
            T = self.mesh.find_field_values_at(self.mesh.left_boundary.coordinates(), self.uh)
            self.mesh.update_boundary_conditions(T, iteration)

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

    def compute_total_heat_flux(self, boundary):
        # heat flux equation: -k * grad(T).n
        integrand = -self.thermal_conductivity *  ufl.dot(ufl.grad(self.uh), self.mesh.n)

        total = fem.assemble_scalar(fem.form(integrand * boundary.ds))
        print(total)
        return total
    
    def compute_heat_flux(self, boundary):
        print(self.mesh.find_field_values_at(boundary.coordinates(), self.uh))
        print(self.mesh.find_field_values_at(boundary.coordinates()+(self.dx*np.array([1.0, 0, 0])), self.uh))

         # Create vector function space for flux
        V_g = fem.functionspace(self.mesh.domain, ("Lagrange", self.poly_order, (self.mesh.gdim, )))

        # Heat flux expression
        flux_expr = -self.thermal_conductivity * ufl.grad(self.uh)

        # Create a function for flux evaluation
        flux = fem.Function(V_g)
        flux.name = "flux"

        # Project flux = -kappa * grad(u) onto V
        w_f = ufl.TrialFunction(V_g)
        v_f = ufl.TestFunction(V_g)
        a_proj = ufl.inner(w_f, v_f) * ufl.dx
        L_proj = ufl.inner(flux_expr, v_f) * ufl.dx
        problem = LinearProblem(a_proj, L_proj)
        flux = problem.solve()

        # Evaluate flux at these dofs
        flux_vals = flux.x.array.reshape((-1, self.mesh.gdim))[boundary.dofs]

        # Gather to root if needed
        left_dof_coords_flux = np.array(boundary.dof_coords)
        flux_vals = np.array(flux_vals)
        normal = np.array([-1, 0.0, 0.0])  # normal vector on the left face at x = -1.0
        flux_vals = np.dot(flux_vals, normal)
        print(self.cellArea)
        print(flux_vals)
   
        # Gather to root if needed
        return flux_vals

    


    def update_flux_from_external(self, t, boundary):
        """Update boundary flux values from external solver."""
        q_flux_array = self.q_flux.x.array
        for i, dof in enumerate(boundary.dofs):
            coord = boundary.dof_coords[i]
            q_flux_array[dof] = coord[0] * coord[1] * coord[2] * t * 0.0 + boundary.fluxValue  # steady flux for testing
        self.q_flux.x.scatter_forward()