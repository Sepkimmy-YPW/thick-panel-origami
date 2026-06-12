# phys_sim_pd14 — Projective Dynamics Origami Folding Simulator

A GPU-accelerated origami folding simulator built on the **Projective Dynamics (PD)** framework, implemented in [Taichi](https://github.com/taichi-dev/taichi). It supports real-time interactive simulation of origami folding with constraint-based mechanics, thick-panel models, and sequential folding control.

## Features

- **Projective Dynamics solver** — implicit time integration with local/global alternation for stable, large-step folding simulation
- **Three constraint types** — spring (in-plane stretching), crease bending (hinge), and facet bending (rigid-panel)
- **Cotangent Laplacian Hessian** — analytically derived 4×4 cotangent weight matrices for bending constraints, assembled into the global system matrix
- **Momentum-conserving dihedral projection** — `project_dihedral_momentum_conserving` uses Discrete Shells gradient formulas (Grinspun et al. 2003) that algebraically satisfy both linear and angular momentum conservation
- **Signed dihedral angle** with barrier-based collision protection — prevents self-intersection near folding angle limits via log-barrier potentials
- **Sequential folding** — level-based folding sequence with configurable per-crease level, coefficient, and recover angles
- **Thick-panel origami** — height-offset decomposition with inter-panel spring connections for simulating rigid-panel (thick) origami
- **Interactive GUI** — Taichi GGUI-based real-time visualization with parameter sliders and keyboard control
- **Headless rendering** — `use_gui=False` mode with `outputFigure()` for batch simulation and image export
- **CDF integration** — designed as the physics backend for Computational Design Framework (CDF) optimization loops

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  PD_Origami_Simulator                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Data &      │  Local Step  │  Global Step │  I/O &         │
│  Topology    │  (Projection)│  (Solve)     │  Rendering     │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ x, v, s      │project_spring│ fill_b()     │ render()       │
│ spring_sel   │project_      │ sparse_      │ outputFigure() │
│ bending_sel  │ bending_2/3  │ solver.solve │ appendCrease   │
│ facet_bend_  │project_facet │ update_x()   │ Info()         │
│ sel          │ bending_2/3  │              │                │
│ cotangent_   │              │              │                │
│ matrix       │              │              │                │
│ masses       │              │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### Simulation Pipeline (per time step)

```
update_folding_target()      ← increment folding angle
         │
    forward(dt)              ← semi-implicit Euler: x_{n+1} = x_n + v_n * dt,  s = x_{n+1}
         │
    ┌──── PD iteration (pd_iter_time times) ────┐
    │                                             │
    │   clearEnergy()                             │
    │        │                                    │
    │   local_step(theta)                         │
    │     ├─ project_spring()       → spring_x_proj
    │     ├─ project_bending_2()    → bending_x_proj
    │     └─ project_facet_bending_2() → facet_bending_x_proj
    │        │                                    │
    │   global_step()                             │
    │     ├─ fill_b(dt)            ← assemble RHS│
    │     ├─ sparse_solver.solve(b)→ u0           │
    │     └─ update_x()           ← x += u0      │
    │                                             │
    └─────────────────────────────────────────────┘
         │
    update_vel(dt)            ← v = (x - x0) / dt * damping
```

### Constraint Models

| Constraint | Projection Kernel | Description |
|---|---|---|
| **Spring** | `project_spring` | Distance constraint preserving edge lengths (in-plane stretching resistance) |
| **Crease bending** | `project_bending_2` / `project_bending_3` | Dihedral angle constraint at crease lines with target angle from sequential folding |
| **Facet bending** | `project_facet_bending_2` / `project_facet_bending_3` | Dihedral angle constraint at facet interior edges, target = flat (θ = 0) |

Two projection methods are available for bending constraints:

- **`_2` variants** — Rodrigues rotation-based projection: rotates free vertices around the crease axis, does not modify crease vertices. Simple but does not conserve angular momentum.
- **`_3` variants** — Momentum-conserving projection via `project_dihedral_momentum_conserving`: uses Discrete Shells dihedral gradient (4-vertex formulation with cotangent weights) to distribute corrections to all four vertices, guaranteeing ∑gᵢ = 0 (linear momentum) and ∑rᵢ × gᵢ = 0 (angular momentum).

## Key Functions Reference

### Construction & Initialization

| Function | Description |
|---|---|
| `__init__(origami_name, use_gui, fast, ...)` | Create simulator instance, allocate Taichi fields and GGUI window |
| `start(filepath, unit_edge_max, thick_mode)` | Load origami description JSON, build topology, call `commonStart_1/2` |
| `commonStart_1(unit_edge_max, thick_mode)` | Construct `OrigamiSimulationSystem`, add units, generate triangulation |
| `commonStart_2()` | Allocate all Taichi fields, assemble constraint indices |
| `initializeRunning()` | Reset simulation state, initialize Taichi fields from NumPy data, compute cotangent weights, factorize Hessian |

### Simulation Loop

| Function | Description |
|---|---|
| `run()` | Main loop: `step()` + `render()` until convergence or window close |
| `step()` | One simulation step: `update_folding_target` → `forward` → PD iterations → `update_vel` |
| `local_step(theta)` | Local projection: `project_spring` + `project_bending_2` + `project_facet_bending_2` |
| `global_step()` | Global solve: `fill_b` → `sparse_solver.solve` → `update_x` |
| `forward(dt)` | Semi-implicit Euler prediction: `x += v * dt` |
| `update_vel(dt)` | Velocity update with damping: `v = (x - x0) / dt * damping` |
| `stop()` | Convergence check: energy decrease + folding angle near π |

### Projection & Force Kernels

| Function | Type | Description |
|---|---|---|
| `project_spring` | `@ti.kernel` | Mass-weighted spring projection |
| `project_bending_2` | `@ti.kernel` | Rodrigues rotation bending projection |
| `project_bending_3` | `@ti.kernel` | Momentum-conserving bending projection |
| `project_facet_bending_2` | `@ti.kernel` | Rodrigues rotation facet bending projection |
| `project_facet_bending_3` | `@ti.kernel` | Momentum-conserving facet bending projection |
| `compute_signed_dihedral` | `@ti.func` | Signed dihedral angle with barrier collision force |
| `project_dihedral_momentum_conserving` | `@ti.func` | Discrete Shells gradient-based dihedral projection |
| `calculateTargetAngle` | `@ti.func` | Compute target angle from sequential folding level |
| `compute_bending_cotangent_weights` | `@ti.kernel` | Build 4×4 cotangent Laplacian matrices for bending Hessian |
| `fill_b` / `fill_b_ndarray` | `@ti.kernel` | Assemble RHS vector for global step |
| `fill_AK_field` | `@ti.kernel` | Assemble global system matrix A = M/h² + K |
| `construct_hessian` | `@ti.kernel` | Build sparse matrix from AK_field |

### Rendering & Output

| Function | Description |
|---|---|
| `render()` | Interactive GUI rendering with parameter sliders and camera control |
| `outputFigure()` | Headless image export (call `window.show()` then `save_image`) |
| `appendCreaseInfo()` | Write final crease angles back to description JSON |
| `reward()` | Compute per-sub-origami reward for CDF optimization |

## Quick Start

### Prerequisites

- Python ≥ 3.9
- Taichi ≥ 1.7
- NumPy

```bash
pip install taichi numpy
```

### Running the Simulator

```python
from phys_sim_pd14 import PD_Origami_Simulator

# Interactive mode
sim = PD_Origami_Simulator("bird4", use_gui=True, fast=False, material_type=1)
sim.start("bird4", 4, thick_mode=0)
sim.run()

# Headless mode (for batch optimization)
sim = PD_Origami_Simulator("bird4", use_gui=False, fast=True, material_type=2)
sim.start("bird4", 4, thick_mode=0)
sim.initializeRunning()
for step in range(max_steps):
    sim.step()
    if sim.stop():
        break
sim.outputFigure()
```

### Command Line

Edit the `__main__` block at the bottom of `phys_sim_pd14.py`:

```python
if __name__ == '__main__':
    ori_name_list = ["bird4"]
    output_fig = 0       # 1 → save frames, 0 → fast mode
    fast_mode = not output_fig

    for ori_name in ori_name_list:
        ori = PD_Origami_Simulator(ori_name, use_gui=True, fast=fast_mode, material_type=1, ref_target=0)
        ori.start(ori_name, 4, thick_mode=0)

    ori.run()
```

Then run:

```bash
python phys_sim_pd14.py
```

### Input Format

The simulator reads origami descriptions from JSON files in `./descriptionData/<name>.json`. Each file contains:

| Field | Description |
|---|---|
| `kps` | List of 2D keypoint coordinates `[[x, y], ...]` |
| `lines` | List of crease segments `[[start_kp, end_kp], ...]` |
| `line_features` | Per-line metadata: `type` (0=border, 1=valley, 2=mountain), `level`, `coeff`, `hard`, `hard_angle`, `hard_angle_down`, `thick_panel_height` |
| `units` | List of unit polygons, each a list of keypoints |
| `contributions` | Per-unit contribution weights (optional) |
| `fix` | List of fixed unit indices |
| `crease_angle` | Target crease angles for `ref_target` mode (optional) |
| `split_num` | Number of sub-origami splits (optional, default 1) |

## Interactive Controls

When `use_gui=True`, the following keyboard shortcuts are available:

| Key | Action |
|---|---|
| `r` | Reset simulation (`initializeRunning`) |
| `u` | Set folding angle to π |
| `j` | Set folding angle to 0 |
| `i` | Start incremental folding (+micro step per frame) |
| `k` | Stop incremental folding |
| `m` | Reverse incremental folding (unfold) |
| `p` | Toggle pause |
| `Space` | Step once (when paused) |
| Right Mouse Button + Drag | Orbit camera |

GUI sliders control `Folding angle`, `Spring k`, `Crease k`, and `Facet k` in real time.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_gui` | `True` | Show interactive window |
| `fast` | `1` | Fast mode (skip frame saving) |
| `pd_iter_time` | `5` | Number of PD local/global iterations per time step |
| `damping` | `0.95` | Velocity damping factor |
| `material_type` | `1` | 1 = standard density (1.24e-9), 2 = light (0.08e-9) |
| `ref_target` | `False` | Scale target angle by per-crease `target_crease_angle` |
| `spring_k` | auto | Spring (stretching) stiffness |
| `bending_k` | auto | Crease bending stiffness |
| `facet_bending_k` | auto | Facet bending stiffness |
| `collision_indice` | `1e-1` | Barrier force magnitude |
| `collision_d` | `1e-4` | Barrier safety distance |

## File Structure

```
phys_sim_pd14.py           # Main simulator (this file)
ori_sim_sys.py             # Origami topology & mesh generation
utils.py                   # Geometry utilities (distance, Crease, Unit classes)
descriptionData/           # Origami JSON descriptions (100+ patterns)
    bird4.json
    miura.json
    auxetic.json
    ...
physResult/                # Output directory for rendered frames
cdf_thick_panel.py         # CDF optimization framework (uses this simulator)
```

## Theoretical Background

- **Projective Dynamics**: Bouaziz et al., "Projective Dynamics: Fusing Constraint Projections for Fast Simulation", ACM TOG (SIGGRAPH) 2014
- **Discrete Shells**: Grinspun et al., "Computing Discrete Shape Operators on General Meshes", Eurographics 2003 — used in `project_dihedral_momentum_conserving` for momentum-conserving dihedral angle gradients
- **Cotangent Laplacian**: Pinkall & Polthier, "Computing Discrete Minimal Surfaces and Their Conjugates", Experimental Mathematics 1993 — used for the 4×4 bending Hessian blocks
- **Log-barrier collision**: incremental barrier method for angle-limit enforcement inspired by IPC (Li et al., SIGGRAPH 2020)

## License

This project is part of the PyGamiX research software suite.
