"""
##############################################################################
# Parallel Partitioned Multi-Physics Simulation Framework (ParaSiF)          #
#                                                                            #
# Copyright (C) 2025 The ParaSiF Development Team                            #
# All rights reserved                                                        #
#                                                                            #
# This software is licensed under the GNU General Public License version 3   #
#                                                                            #
# ** GNU General Public License, version 3 **                                #
#                                                                            #
# This program is free software: you can redistribute it and/or modify       #
# it under the terms of the GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or          #
# (at your option) any later version.                                        #
#                                                                            #
# This program is distributed in the hope that it will be useful,            #
# but WITHOUT ANY WARRANTY; without even the implied warranty of             #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
# GNU General Public License for more details.                               #
#                                                                            #
# You should have received a copy of the GNU General Public License          #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.      #
##############################################################################

    @file thermalStructure.py

    @author W. Liu

    @brief FEniCSx transient heat conduction solver for conjugate heat transfer
           coupling with backward Euler time stepping.

"""

# -------------------------
#%% Import general packages
# -------------------------

import numpy as np
from scipy.spatial import cKDTree

# -------------------------
#%% Control parameters
# -------------------------

quiet = True # define quiet mode
debug = True # define debug mode

iMUICoupling = False  # whether to enable MUI coupling
synchronised=False   # synchronised for announce span
rMUIFetcher = 1.0    # MUI seatch redius
push_coord_dtype = np.float32 # numpy data types for push coordinates
push_flux_dtype = np.float64  # numpy data types for push values

t0 = 0.0  # initial time
T = 70.0  # final time
num_steps = 70/0.05 # number of time steps
num_iterations = 1 # number sub-iterations per time step

kappa_val = 1.0 #thermal diffusivity

nx, ny, nz = 10, 10, 1  # mesh divisions in x,y,z directions
p_min = np.array([-0.1, -0.5, -0.005], dtype=np.float64)  # domain min corner
p_max = np.array([ 0.1,  0.5,  0.005], dtype=np.float64)  # domain max corner

poly_order = 2 # finite element polynomial order

LEFT_MARK = 88  # marker for left boundary facets
RIGHT_MARK = 66  # marker for right boundary facets

# Define the boundary: left face at x = -1.0
def boundary_left(x):
    return np.isclose(x[0], -0.1)

# Define the boundary: right face at x = 1.0
def boundary_right(x):
    return np.isclose(x[0], 0.1)

T0 = 273.15     # baseline temperature
T_L = 773.15 # initial left boundary temperature
T_R = 273.15    # initial right boundary temperature
q_R = 0.0    # Constant heat flux with unit of W/m² as Neumann BC value at right end

def line_mask(dof_coords_output): # mask for dofs along a 1-D line for ASCII output
    return (np.isclose(dof_coords_output[:, 1], 0.0, atol=1e-8) & 
            np.isclose(dof_coords_output[:, 2], 0.0, atol=1e-8))

xdmf_filename = "fenicsx_solid_coupled.xdmf" # XDMF output filename
line_coords_filename = "line_dofs_coords.dat" # line dof coordinates output filename
timeseries_filename = "dof_temperature_timeseries.dat" # temperature timeseries output filename


interfaceName = "ifs1"

# -------------------------
#%% initialise MUI/MPI for coupling
# -------------------------
from mpi4py import MPI
import petsc4py

if iMUICoupling:
    import mui4py
  
    # MUI parameters
    dimensionMUI = 3
    data_types = {"temp": mui4py.FLOAT64,
                  "flux": mui4py.FLOAT64}
    # MUI interface creation
    domain = "structureDomain"
    config3d = mui4py.Config(dimensionMUI, mui4py.FLOAT64)

    # App common world claims
    LOCAL_COMM_WORLD = mui4py.mpi_split_by_app(argc=0,
                                            argv=[],
                                            threadType=-1,
                                            thread_support=0,
                                            use_mpi_comm_split=True)

    iface = [interfaceName]
    ifaces3d = mui4py.create_unifaces(domain, iface, config3d)
    ifaces3d[interfaceName].set_data_types(data_types)

else:
    LOCAL_COMM_WORLD = MPI.COMM_WORLD

# Necessary to avoid hangs at PETSc vector communication
petsc4py.init(comm=LOCAL_COMM_WORLD)
# Define local communicator rank
rank = LOCAL_COMM_WORLD.Get_rank()
# Define local communicator size
size = LOCAL_COMM_WORLD.Get_size()

# -------------------------
#%% Import FEniCSx packages
# -------------------------

from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc, LinearProblem

# -------------------------
#%% Problem / time parameters
# -------------------------

t = t0 # time
dt = (T - t0) / num_steps # time step size

# -------------------------
#%% Create mesh and function space
# -------------------------

domain = mesh.create_box(LOCAL_COMM_WORLD,
                         [p_min, p_max],
                         [nx, ny, nz],
                         cell_type=mesh.CellType.hexahedron,
                         ghost_mode=mesh.GhostMode.shared_facet)

gdim = domain.geometry.dim
# Define a poly_order function space for simulation
V = fem.functionspace(domain, ("Lagrange", poly_order))
# Define a P1 function space for output
V_out = fem.functionspace(domain, ("Lagrange", 1))
kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))

# -------------------------
#%% Identify boundary facets and coupled facets
# -------------------------

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# exterior facets indices (global entity indices)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# all boundary DOFs
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

if not quiet:
    print("{{FENICS}} Rank", LOCAL_COMM_WORLD.rank, "Number of boundary dofs:", len(boundary_dofs), "boundary facets:", len(boundary_facets))

left_facets = mesh.locate_entities_boundary(domain, fdim, boundary_left)

# marker id for the left boundary
left_facet_tag = mesh.meshtags(domain, fdim, left_facets, np.full_like(left_facets, LEFT_MARK))

# left DOFs for Dirichlet BC
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
if not quiet:
    print("{{FENICS}} Rank", LOCAL_COMM_WORLD.rank, "Number of left dofs:", len(left_dofs))

# Extract coordinates of DOFs
dof_coords = V.tabulate_dof_coordinates().reshape((-1, gdim))
left_dof_coords = dof_coords[left_dofs]

right_facets = mesh.locate_entities_boundary(domain, fdim, boundary_right)

# marker id for the right boundary
right_facet_tag = mesh.meshtags(domain, fdim, right_facets, np.full_like(right_facets, RIGHT_MARK))

# right DOFs
right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
if not quiet:
    print("{{FENICS}} Rank", LOCAL_COMM_WORLD.rank, "Number of left dofs:", len(right_dofs))

# Extract coordinates of DOFs
right_dof_coords = dof_coords[right_dofs]

# -------------------------
#%% Function to hold Dirichlet values
# -------------------------

u_bc = fem.Function(V)
u_bc.name = "u_bc"

# Create left boundary condition
class boundary_condition():
    def __init__(self, t, coupled_dof_coords):
        self.t = t
        self.tree = cKDTree(coupled_dof_coords)  # build KDTree for coupled DOF coords

    def update_time(self, t):
        """Update current simulation time before interpolation."""
        self.t = t
        self.steps = int(round((t - t0) / dt))  # update time step

    def __call__(self, x):
        tol = 1e-8
        values = np.zeros(x.shape[1])
        # Query nearest boundary point distance for each x
        dist, _ = self.tree.query(x.T)
        on_boundary = dist < tol
        # Use only the boundary coordinates
        x_boundary = x[:, on_boundary]
        dofs_boundary = x_boundary.T
        if iMUICoupling and (abs(t - t0) > 1e-12):
            values[on_boundary] = ifaces3d[interfaceName].\
                                    fetch_many("temp",
                                                dofs_boundary,
                                                self.t,
                                                s_sampler,
                                                t_sampler)
            if (not quiet) and debug:
                print("Boundary values:", values[on_boundary], " Dof Boundary: ", dofs_boundary, " self.steps: ", self.steps)
        else:
            values[on_boundary] = (x[0, on_boundary] *
                                   x[1, on_boundary] *
                                   x[2, on_boundary] * 0.0) + T_L  # steady 500K for testing
        return values

u_boundary = boundary_condition(t, left_dof_coords)
u_boundary.update_time(t)
u_bc.interpolate(u_boundary)

bc_left = fem.dirichletbc(u_bc, left_dofs)
bc_right = fem.dirichletbc(u_bc, right_dofs)
bc = [bc_left, bc_right]

# -------------------------
#%% Function to receive heat flux across the right boundary
# -------------------------

ds_right = ufl.Measure("ds", domain=domain, subdomain_data=right_facet_tag)
# FacetNormal(domain) yields the outward normal on the domain boundary facets
n = ufl.FacetNormal(domain)

q_flux = fem.Function(V)
q_flux.name = "q_flux"

area_right = (p_max[1] - p_min[1]) * (p_max[2] - p_min[2])
if (not quiet) and debug:
    print("{{FENICS}} Right boundary area:", area_right)
q_R_total = q_R * area_right
if (not quiet) and debug:
    print("{{FENICS}} Total heat flux at right boundary (W):", q_R_total)
q_R_per_dof = q_R_total / len(right_dofs)
if (not quiet) and debug:
    print("{{FENICS}} Heat flux per right boundary DOF (W):", q_R_per_dof)

def update_flux_from_external(t):
    """Update boundary flux values from external solver."""
    q_flux_array = q_flux.x.array
    for i, dof in enumerate(right_dofs):
        coord = right_dof_coords[i]
        q_flux_array[dof] = coord[0] * coord[1] * coord[2] * t * 0.0 + q_R_per_dof  # steady flux for testing
    q_flux.x.scatter_forward()

def compute_total_right_heat_flux(func_u):
    integrand = -kappa * ufl.dot(ufl.grad(func_u), n)
    total = fem.assemble_scalar(fem.form(integrand * ds_right(RIGHT_MARK)))
    # assemble_scalar returns a scalar on each rank; convert to global sum
    total_global = LOCAL_COMM_WORLD.allreduce(total, op=MPI.SUM)
    return float(total_global)

update_flux_from_external(0.0)

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.x.array[:] = T0

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, PETSc.ScalarType(0.0))  # source term

F = (u - u_n) / dt * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx - q_flux * v * ds_right(RIGHT_MARK)
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

# Assemble matrix with Dirichlet BCs applied
A = assemble_matrix(a, bcs=bc)
A.assemble()
b = create_vector(L)

# Function for current solution at each time step
uh = fem.Function(V)
uh.name = "uh"
uh.x.array[:] = T0

# Function for output at each time step
uh_out = fem.Function(V_out)
uh_out.interpolate(uh)

# Set up PETSc KSP solver
solver = PETSc.KSP().create(LOCAL_COMM_WORLD)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# -------------------------
#%% Setup XDMF output
# -------------------------

xdmf = io.XDMFFile(LOCAL_COMM_WORLD, xdmf_filename, "w")
xdmf.write_mesh(domain)

# -------------------------
#%% Function to compute heat flux across the left boundary
# -------------------------

# Create UFL measure on boundary with subdomain_data
ds_left = ufl.Measure("ds", domain=domain, subdomain_data=left_facet_tag)

def compute_total_left_heat_flux(func_u):
    integrand = -kappa * ufl.dot(ufl.grad(func_u), n)
    total = fem.assemble_scalar(fem.form(integrand * ds_left(LEFT_MARK)))
    # assemble_scalar returns a scalar on each rank; convert to global sum
    total_global = LOCAL_COMM_WORLD.allreduce(total, op=MPI.SUM)
    return float(total_global)

# Compute local heat flux vector and coordinates at each DOF on the coupled boundary.
def compute_heat_flux_on_left_boundary(t, u, kappa):
    # Create vector function space for flux
    V_g = fem.functionspace(domain, ("Lagrange", poly_order, (domain.geometry.dim, )))

    # Heat flux expression
    flux_expr = -kappa * ufl.grad(u)

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
    flux_vals = flux.x.array.reshape((-1, gdim))[left_dofs]

    # Gather to root if needed
    left_dof_coords_flux = np.array(left_dof_coords, dtype=push_coord_dtype)
    flux_vals = np.array(flux_vals, dtype=push_flux_dtype)

    if iMUICoupling:
        ifaces3d[interfaceName].push_many("flux", left_dof_coords_flux, flux_vals[:, 0])
        #ifaces3d[interfaceName].push_many("heatFluxy", left_dof_coords_flux, flux_vals[:, 1])
        #ifaces3d[interfaceName].push_many("heatFluxz", left_dof_coords_flux, flux_vals[:, 2])
        ifaces3d[interfaceName].commit(t)
        if not quiet:
            print('{{FENICS}} MUI commit step: ',int(round((t - t0) / dt)))
            if debug:
                print('{{FENICS}} Push at: ', left_dof_coords_flux, ' flux_vals[:, 0]: ', flux_vals[:, 0], ' at ', int(round((t - t0) / dt)))

    # Print flux values and coordinate components
    if not quiet and debug:
        print("{{FENICS}} Heat flux vectors at coupled boundary DOFs:")
        for coord, flux_vec in zip(left_dof_coords_flux, flux_vals):
            x, y, z = coord  # unpack coordinate components
            print(f"{{FENICS}} x = {x:.6f}, y = {y:.6f}, z = {z:.6f} | Flux = [{flux_vec[0]:.6e}, {flux_vec[1]:.6e}, {flux_vec[2]:.6e}]")

    # Gather to root if needed
    left_dof_coords_flux = np.vstack(LOCAL_COMM_WORLD.allgather(left_dof_coords_flux))
    flux_vals = np.vstack(LOCAL_COMM_WORLD.allgather(flux_vals))

    # Compute total flux as sum of normal components at coupled DOFs
    total_flux = 0.0
    for coord, flux_vec in zip(left_dof_coords_flux, flux_vals):
        normal = np.array([-1.0, 0.0, 0.0])  # normal vector on the left face at x = -1.0
        total_flux += np.dot(flux_vec, normal)

    return total_flux

# -------------------------
#%% Select DoFs for ASCII output
# -------------------------

dof_coords_output = V.tabulate_dof_coordinates()
line_dofs = np.where(line_mask(dof_coords_output))[0]
line_coords = dof_coords_output[line_dofs,0]
sort_idx = np.argsort(line_coords)
line_dofs = line_dofs[sort_idx]
line_coords = line_coords[sort_idx]

if LOCAL_COMM_WORLD.rank == 0:
     # Save DOF coordinates along the selected line
    np.savetxt(line_coords_filename, line_coords)
    # Open ASCII file for temperature time series
    f_ascii = open(timeseries_filename, "w")
    header = "# time " + " ".join([f"x={x:.4f}" for x in line_coords])
    f_ascii.write(header + "\n")

# -------------------------
#%% Define MUI samplers and commit ZERO step
# -------------------------

if iMUICoupling:
    if len(dof_coords) != 0:
        domain_mins = np.min(dof_coords, axis=0)
        domain_maxs = np.max(dof_coords, axis=0)

        if np.any(domain_maxs < domain_mins):
            print(f"{{** FENICS ERROR **}} Invalid bounding box at rank {rank}")

        span = mui4py.geometry.Box(domain_mins, domain_maxs)

        # Announce MUI send span
        ifaces3d[interfaceName].announce_send_span(0, num_steps*num_iterations, span, synchronised)
        ifaces3d[interfaceName].announce_recv_span(0, num_steps*num_iterations, span, synchronised)

        print(f"{{FENICS}} rank {rank} send_recv_min: {domain_mins}, send_recv_max: {domain_maxs}")

    t_sampler = mui4py.TemporalSamplerExact()
    s_sampler = mui4py.SamplerPseudoNearestNeighbor(rMUIFetcher)

    # Commit ZERO step before time stepping
    ifaces3d[interfaceName].commit(0)
    print(f"{{FENICS}} Commit ZERO step")

# -------------------------
#%% Time-stepping loop
# -------------------------

for step in range(1, int(num_steps) + 1):
    t += dt
    print(f"{{FENICS}} [rank {LOCAL_COMM_WORLD.rank}] Time step {step}/{num_steps}, t = {t:.6f}")

    # Update Diriclet boundary condition
    u_boundary.update_time(t)
    u_bc.interpolate(u_boundary)

    # Update heat flux
    update_flux_from_external(t)
    total_flux_right = compute_total_right_heat_flux(uh)
    if not quiet:
        if LOCAL_COMM_WORLD.rank == 0:
            # positive flux means heat leaving the solid across the boundary (sign follows -kappa*grad·n)
            print(f"{{FENICS}}  Total heat flux across right boundary (integral) = {total_flux_right:.6e}")

    # Assemble RHS with current u_n
    with b.localForm() as loc_b:
        loc_b.set(0.0)
    assemble_vector(b, L)

    # Apply BCs to RHS
    apply_lifting(b, [a], [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bc)

    # Solve linear system A * uh = b
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()


    # Compute heat flux across the left boundary
    total_flux_push = compute_heat_flux_on_left_boundary(t, uh, kappa)
    total_flux = compute_total_left_heat_flux(uh)
    if not quiet:
        if LOCAL_COMM_WORLD.rank == 0:
            # positive flux means heat leaving the solid across the boundary (sign follows -kappa*grad·n)
            print(f"{{FENICS}}  Total heat flux across left boundary (integral) = {total_flux:.6e}; and total heat flux push (DOF sum) = {total_flux_push:.6e}")

    # Update u_n for next time step
    u_n.x.array[:] = uh.x.array

    # Write solution to XDMF file
    uh_out.interpolate(uh)
    xdmf.write_function(uh_out, t)

    print(uh)
    print(uh.x.array.shape)

    # Write ASCII temperature timeseries at selected line DOFs
    if LOCAL_COMM_WORLD.rank == 0:
        line_temps = uh.x.array[line_dofs]
        f_ascii.write(f"{t:.6f} " + " ".join(f"{temp:.6f}" for temp in line_temps) + "\n")

# -------------------------
#%% Finalise
# -------------------------

# Close ASCII file after time-stepping
if LOCAL_COMM_WORLD.rank == 0:
    f_ascii.close()
# Close XDMF file after time-stepping
xdmf.close()
# Indicate run completion
if LOCAL_COMM_WORLD.rank == 0:
    print("{{FENICS}} Run complete.")

if iMUICoupling:
    ifaces3d[interfaceName].barrier(-888)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%  FILE END  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
