from mpi4py import MPI
import mui4py


LOCAL_COMM_WORLD = MPI.COMM_WORLD
PYTHON_COMM_WORLD = mui4py.mpi_split_by_app()

import petsc4py
petsc4py.init(comm=PYTHON_COMM_WORLD)

from output import Output
from heatEquationFenics import HeatEquationFenics


heatSolver = HeatEquationFenics(PYTHON_COMM_WORLD, LOCAL_COMM_WORLD)

dof_coords_output = heatSolver.V.tabulate_dof_coordinates()
output = Output(heatSolver.mesh.domain, dof_coords_output, PYTHON_COMM_WORLD)

t = 0
c = 1
writeIntervalSteps = int(heatSolver.writeInterval / heatSolver.dt)
for step in range(1, heatSolver.num_steps + 1):
    t += heatSolver.dt
    for i in range(heatSolver.inner_loop_iterations):
        heatSolver.update_boundary_conditions(c)
        heatSolver.solve()
        c += 1   
    
    heatSolver.update_time()
    if step % writeIntervalSteps == 0:
        output.writeFunction(heatSolver.uh_out, t)

output.close()

