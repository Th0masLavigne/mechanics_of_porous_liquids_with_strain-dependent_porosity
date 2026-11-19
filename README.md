# Porous Media Finite Element Simulations

A collection of **Finite Element Method (FEM)** simulations developed using **DolfinX** to model problems involving **porous media mechanics** for **porous liquid** formulations. The project is structured into two main simulation cases: a channel flow problem and a transient consolidation problem.

These codes accompany the article *Mechanics of porous liquids with strain-dependent porosity* (G. Sciumè, T. Lavigne, W. G. Gray), currently **under review**.

---

## 📖 Citation

If you use these codes, or part of them, in your work, **please cite the related article**:

> Sciumè, G., Lavigne, T., & Gray, W. G. *Mechanics of porous liquids with strain-dependent porosity*. Submitted (under review).

```bibtex
@unpublished{Sciume_Lavigne_Gray_porous_liquids,
  author = {Sciumè, Giuseppe and Lavigne, Thomas and Gray, William G.},
  title  = {Mechanics of porous liquids with strain-dependent porosity},
  note   = {Submitted, under review},
  year   = {2026}
}
```

*This reference will be updated with the journal, volume and DOI once the article is published.*

---

## Project Structure

The core of the project is organized into two main directories, each representing a distinct simulation case. Utility scripts shared across simulations are located within the `service_files_*` directories.

```text
.
├── Channel/
│       ├── Croix_porous_liquid.py
│       ├── run_convergence_study.sh
│       ├── convergence_postprocess.py
│       └── service_files_channel/
└── Consolidation/
        ├── Porous_Liquid.py
        └── service_files_consolidation/
```

-----

## 🛠️ Prerequisites

The simulations rely on the **FEniCS Project** ecosystem. The following dependencies are required:

  * **Python 3.x**
  * **FEniCSx** (The core Finite Element library)
  * **mpi4py** (For parallel execution)
  * **Gmsh** (Used for mesh generation)
  * **Pandas**

-----
## Docker environment

FEniCSx v0.9.0:

```cmd
docker run -ti -v $(pwd):/home/fenicsx/shared -w /home/fenicsx/shared th0maslavigne/dolfinx:v0.9.0
```

-----

## Simulation Cases

### 1\. Channel Flow Simulation (`Channel/`)

This case simulates fluid flow through a domain, including a T-junction (dead-end), using a porous liquid model.

| File/Folder | Description |
| :--- | :--- |
| `Croix_porous_liquid.py` | **Main execution script** for setting up and running the channel flow simulation. |
| `run_convergence_study.sh` | Runs the spatial (mesh) and temporal (time-step) convergence study. |
| `convergence_postprocess.py` | Computes the convergence metrics and writes the csv files used in the Supplementary Material (Table S1, Figure S1). |
| `service_files_channel/` | Contains the utility modules. |

`Croix_porous_liquid.py` accepts optional command-line arguments (defaults reproduce the article: `--dt 1e-2 --size_max 1e-3`):

```bash
mpirun -n 4 python3 Croix_porous_liquid.py --dt 5e-3 --size_max 5e-4 --tag _test --no_fields
```

**Convergence study** (from `Channel/`):

```bash
NP=4 bash run_convergence_study.sh      # 4 mesh sizes (dt = 0.01 s) + 5 time steps (h_max = 1 mm)
python3 convergence_postprocess.py      # -> convergence_results/*.csv, summary and quick-look figure
```

The metric is the relative L2 error of the axial velocity profiles at x = 0.1L and x = 0.9L (accumulated over all time steps, and at t = 0.01, 1.15, 5 s), plus the error on the infiltration velocity at the branch mid-point, with respect to the finest discretization of each series. Observed orders of convergence are also reported.

### 2\. Consolidation Simulation (`Consolidation/`)

This case focuses on the transient problem of **consolidation**, modeling the dissipation of fluid pressure under load in a porous medium. The geometry is likely an axisymmetric rectangle.

| File/Folder | Description |
| :--- | :--- |
| `Porous_Liquid.py` | **Main execution script** for solving the time-dependent consolidation problem. |
| `service_files_consolidation/` | Contains the utility modules. |

-----

## Utility Modules

The following Python modules define core components and are utilized by both main simulation scripts (`Croix_porous_liquid.py` and `Porous_Liquid.py`):

| File | Description |
| :--- | :--- |
| `constitutive_laws.py` | Defines material-specific **constitutive laws**, including functions for computing coefficients (e.g., Byrne coefficients) for the **cohesive zone model** applied to porous media. |
| `functions.py` | Contains general **DolfinX utility functions**, such as solver configuration settings, numerical differentiation functions, and specialized **Axisymmetric operators** (e.g., `grad_cyl_x`, `div_cyl_x`). |
| `geometry_porous_liquid.py` | Provides Python functions (e.g., `Axysimmetric_rectangle_mesh`) for programmatically generating the **Gmsh geometries** and defining boundary conditions tags for the different simulation domains. |

-----

## How to Run a Simulation

1.  **Navigate** to the desired simulation directory:

    ```bash
    # For the Channel simulation
    cd Channel/ 
    # OR
    # For the Consolidation simulation
    cd Consolidation/
    ```

2.  **Execute** the main Python script using `mpiexec` for parallel processing (recommended for performance with DolfinX):

    ```bash
    # Run Channel flow simulation on 4 cores
    mpirun -n 4 python3 Croix_porous_liquid.py

    # Run Consolidation simulation on 4 cores
    mpirun -n 4 python3 Porous_Liquid.py
    ```

-----

*Note: Any output files, mesh logs, or temporary caches generated by the scripts are ignored in this documentation.*
