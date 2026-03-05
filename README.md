# heatSolverFenicsX

A Python-based finite element solver for the heat equation problems using the FEniCSx ecosystem, including `dolfinx`, `ufl`, and PETSc-backed linear solvers. This solver is coupled with Openfoam as a part of the ParaSiF partitioned solver for Thermal-Fluid-Structure Interaction (TFSI) using the Multiscale Universal Interface (MUI).
---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Visualisation](#visualisation)

## Project Structure
```
heatSolverFenicsX
│
├──src
  ├── solver.py              # Main solver containing outer and inner loops
  ├── heatEquationFenics     # Definition of variational form of heat eqation, creation of function spaces and PETSc solvers
  ├── meshGeneration.py      # Creation of mesh topology
  ├── boundarys.py           # Various BC definitions, includuding CoupledBoundaries responsible for MUI coupling
  ├── input.py               # Responsible for parsing case files; case.json and solver.json.
  ├── output.py              # Responsible for exporting mesh and field data to xdmf format for ParaView
├── README.md
├── originalCode.py          # Orginal code written by Wendi Liu from which this work is derived.
└── spack.yaml               # Spack environment dependencies

```

# Requirements

The project depends on the **FEniCSx** finite element ecosystem and mui4py (MUI Python Wrappers). The code has been tested using Python 3.12.

Required packages:
- `scipy v1.16.3`
- `dolfinx v0.9.0`
- `ufl v2024.2.0`
- `basix v0.9.0`
- `mpi4py v4.1.1`
- `petsc4py v3.24.3`
- `numpy v2.3.5`
- `mui4py @ master ([branch](https://github.com/MxUI/MUI))`
  
---

# Installation

The best way to get started is using [spack](https://spack.io/). Using the **spack.yaml** file run the following with an environment name of your choice.

```bash
spack env create <environment name> spack.yaml
spack env activate -p <environment name>
spack install   # this will take a while
```

---

# Usage

Run the heat solver:

```bash
python solver.py
```

For MPI parallel execution:

```bash
mpirun -n 4 python solver.py
```

Outputs such as temperature fields may be written to:

```
results/
```

for visualisation.

---

# Example Workflow

1. Define a computational mesh
2. Specify boundary conditions
3. Define the variational form of the PDE
4. Assemble the finite element system
5. Solve using PETSc linear solvers
6. Export the solution

Example simplified code structure:

```python
from dolfinx import mesh, fem
import ufl

# Create mesh
domain = mesh.create_unit_square(...)

# Define function space
V = fem.FunctionSpace(domain, ("Lagrange", 1))

# Define weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ...

# Solve system
```

---

# Visualisation

Solutions can be visualised using:

- **PyVista**
- **ParaView**

Example output formats:

- `.xdmf`
- `.vtu`

ParaView can directly load XDMF files for interactive visualisation.

---

# Extending the Solver

Possible improvements:

- Time-dependent heat equation
- Adaptive mesh refinement
- Variable conductivity materials
- Multi-physics coupling
- GPU acceleration via PETSc

---

# Development

This project is intended as a **learning and experimentation platform** for:

- Finite element methods
- Scientific computing
- PDE solvers
- The FEniCSx ecosystem

---

# Contributing

Contributions are welcome.

Possible contributions include:

- Additional example problems
- Performance improvements
- Documentation improvements
- Additional physics models

Submit a pull request or open an issue.

---

# License

Please refer to the repository for license information.

If no license file is included, the code should be assumed to be **all rights reserved** until a license is added.
