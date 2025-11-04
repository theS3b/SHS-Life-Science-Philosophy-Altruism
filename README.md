# 🧬 CUDA Grid Simulation for Evolutionary Dynamics & Altruism

<font size=4><div align='center'>
[[📄 Tech & Philosophical Report (French - Original)](pdf/(Original)%20Etude%20des%20conditions%20d'émergence%20de%20l'altruisme%20par%20le%20biais%20de%20simulations%20informatiques.pdf)]<br>
[[📄 Tech & Philosophical Report (English - Translation)](pdf/(Translation)%20Study%20of%20the%20Conditions%20for%20the%20Emergence%20of%20Altruism%20through%20Computer%20Simulations.pdf)]
</div></font>


### 📝 Abstract

*This study investigates the emergence of altruism through a computational model of agents interacting on a discrete 20×20 grid. Three types of agents were simulated: random, semi-random (giving or attacking), and intelligent agents trained with reinforcement learning. By varying two key parameters, the donation bonus and growth rate, the model tests how environmental conditions influence cooperation and survival. 
Results show that altruism appears only under specific conditions. Giving agents prevail with a high donation bonus, while attacking agents dominate in growing environments. Intelligent agents adapt by exploiting or cooperating depending on their opponents, balancing self-interest and collective benefit.
Although the agents lack intention or emotion and cannot display psychological altruism, the model reproduces functional and behavioral forms of cooperation. It provides a simple but extensible framework to study how basic interaction rules can lead to the emergence of complex social behavior.*


### 📚 Technical Introduction

We present a CUDA-accelerated grid simulation framework designed to investigate the emergence and evolution of altruistic behavior in multi-agent populations. The simulation models three competing populations on a toroidal grid, where agents can engage in both cooperative (altruistic donation) and competitive (attack) behaviors. Each cell maintains a fitness value that determines colonization success and survival outcomes. By leveraging PyTorch's GPU acceleration with circular convolution operations, the framework can simulate thousands of parallel environments simultaneously, enabling large-scale studies of evolutionary strategies under various selective pressures. The system is wrapped with a Gymnasium-compatible interface for reinforcement learning experiments, allowing researchers to train agents that learn optimal cooperation-competition strategies.

## 🎯 Project Overview

This project investigates the evolutionary dynamics of altruism through massively parallel grid-based simulations. The core simulation (`cuda_square_simulation.py`) uses GPU-accelerated tensor operations to model population interactions across thousands of parallel worlds, making it ideal for:

- **Evolutionary Biology Research**: Study how altruistic behaviors emerge and persist under different selection pressures
- **Multi-Agent Reinforcement Learning**: Train agents to learn cooperation strategies in competitive environments
- **Life Science Philosophy**: Explore the conditions under which self-sacrifice for group benefit becomes evolutionarily stable

### Key Features

🚀 **GPU-Accelerated**: Simulate 10,000+ parallel environments simultaneously using CUDA  
🧬 **Population Dynamics**: Three competing populations with configurable fitness parameters  
🤝 **Altruistic Behaviors**: Agents can donate fitness to neighbors at personal cost  
⚔️ **Conflict Resolution**: Combat mechanics with fitness redistribution to survivors  
🔄 **Toroidal Grid**: Circular boundary conditions for spatially continuous dynamics  
🎮 **RL-Ready**: Gymnasium-compatible environment for training intelligent agents  
📊 **Real-Time Visualization**: Interactive matplotlib-based rendering with playback controls

## 🧮 Simulation Mechanics

### Core Dynamics

The simulation operates in discrete time steps with two main phases:

#### 1. Colonization Phase
Empty cells can be colonized by neighboring populations. The probability of successful colonization is proportional to the average fitness of surrounding cells of that population:

```python
colonization_prob = neighbor_avg_fitness / COLONIZE_PROB_ONE
```

Where `COLONIZE_PROB_ONE = 2.0` (fitness value for 100% colonization probability).

#### 2. Conflict Phase
Occupied cells can perform actions (controlled by agents or random):

**Actions 0**: Do nothing (rest)

**Actions 1-8**: Altruistic donation to neighbors
- Donate 10% of fitness to a neighboring cell (8 directions)
- Donor loses 10% fitness (costly altruism)
- ⚠️ **Warning**: Donating to empty cells results in pure fitness loss

**Actions 9-16**: Attack neighbors  
- Attack a neighboring cell (8 directions)
- Both attacker and defender lose `min(attacker_fitness, defender_fitness)`
- If either cell dies (fitness ≤ 0), its remaining fitness is redistributed equally to all surviving neighbors
- Death creates potential colonization opportunities

### Population Parameters

Each population is defined by:
- `p`: Initial colonization probability (e.g., 0.2 = 20% of grid)
- `mean_v`: Mean initial fitness value (e.g., 1.0)
- `std_v`: Standard deviation of fitness values (e.g., 0.2)

### Technical Implementation

The simulation uses **depth-wise circular convolution** for efficient parallel computation:
- Colonization: Weighted neighbor averaging via `F.conv2d` with `CircularPad2d`
- Attacks: Directional `torch.roll` operations for conflict resolution
- Donations: Vectorized fitness transfer with masked updates

All operations are batched across populations and environments for maximum GPU utilization.

## 🛠️ Setup

For reproducible environments, we use a conda-compatible tool called **Pixi**.

### Install Pixi

If you don't have Pixi installed, run:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Or on Windows (PowerShell):

```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

### Install the Environment

Navigate to the project directory and install dependencies:

```bash
cd "Grid Simulation/cuda_env_project"
pixi install
```

You can enter the environment with:

```bash
pixi shell
```

Or prepend `pixi run` to individual commands:

```bash
pixi run python cuda_square_simulation.py
```

### System Requirements

- **CUDA 12.2** or higher
- **Python 3.13**
- **PyTorch** (with CUDA support)
- **8GB+ GPU memory** recommended for large batch sizes

### Verify CUDA Installation

Check that CUDA is working correctly:

```bash
pixi run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Run a Basic Simulation

```python
import torch
from cuda_square_simulation import SquareSimulation, random_initial_grid

# Configuration
nb_batches = 1000  # Number of parallel simulations
rows, cols = 50, 50
populations = {
    "red": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
    "blue": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
    "green": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simulation = SquareSimulation(
    nb_batch=nb_batches, 
    rows=rows, 
    cols=cols, 
    populations=populations, 
    device=device
)

# Initialize with random populations
random_initial_grid(simulation, populations, nb_batches, rows, cols, device)

# Run simulation
for _ in range(100):
    action_grid = torch.randint(0, 17, (nb_batches, rows, cols), device=device)
    simulation.step(action_grid)

simulation.print_grid()  # Display first batch
```

### Run with Visualization

```bash
pixi run python "Grid Simulation.py"
```

This launches an interactive visualization with controls:
- **Run**: Start continuous simulation
- **Next Step**: Advance one time step
- **Pause**: Pause simulation

## 💻 Project Breakdown

See the folder `Grid Simulation/cuda_env_project` for the core implementation.

### Core Simulation (`cuda_square_simulation.py`)

The heart of the project: a fully vectorized, GPU-accelerated simulation engine.

**Key Classes:**
- `SquareSimulation`: Main simulation class with batched grid operations
  - `colonization_phase()`: Empty cell colonization via neighbor averaging
  - `conflict_phase()`: Agent actions (donations and attacks)
  - `manage_donations()`: Altruistic fitness transfers
  - `manage_attacks()`: Combat with fitness redistribution


## 📊 Research Applications

### Evolutionary Biology

Study conditions for the evolution of altruism:
1. **Kin Selection**: Does spatial clustering lead to altruism?
2. **Group Selection**: Can inter-population competition favor cooperation?
3. **Reciprocal Altruism**: Do agents learn tit-for-tat strategies?

### Reinforcement Learning

Train agents to discover optimal strategies:
- Balance exploration (attacks) vs. exploitation (colonization)
- Learn when altruism benefits long-term fitness
- Emergent coordination without explicit communication

### Philosophy of Altruism

Explore foundational questions raised by the study:

* Under what environmental conditions can altruistic behavior emerge?
* Can altruistic agents survive when competing with selfish or aggressive populations?
* How does the structure of the environment shape cooperation and collective survival?


---

<div align='center'>

**Questions or suggestions?** Open an issue or start a discussion!

</div>

