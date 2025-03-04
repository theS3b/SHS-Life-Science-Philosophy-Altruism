import random
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button


# --------------------------
# Simulation code
# --------------------------
class HexSimulation:
    def __init__(self, rows, cols, populations, initial_grid=None):
        """
        :param rows: number of rows in the grid.
        :param cols: number of columns in the grid.
        :param populations: dict mapping population id to parameters, e.g.
                            {"red": {"p": 0.8, "mean_v": 1.0, "std_v": 0.2}, ... }
        :param initial_grid: optional initial grid (2D list). If None, grid is empty.
        """
        self.rows = rows
        self.cols = cols
        self.populations = populations  # population parameters
        if initial_grid is not None:
            self.grid = initial_grid
        else:
            self.grid = [[None for _ in range(cols)] for _ in range(rows)]
    
    def get_neighbors(self, r, c, border_connected=True):
        """
        Returns valid neighbor coordinates using an 'even-r' style offset,
        though the exact offsets may differ from the typical pointy-top standard.
        Adjust if you prefer a canonical pointy-top or flat-top layout.
        """
        if r % 2 == 0:
            # Even row offsets
            directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            # Odd row offsets
            directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if border_connected:
                nr, nc = nr%self.rows, nc%self.cols
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def colonization_phase(self):
        """Each empty cell can be colonized by a neighboring population."""
        new_cells = []  # list of updates: (r, c, new_cell)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] is None:
                    # Look at neighbors
                    neigh_coords = self.get_neighbors(r, c)
                    pop_neighbors = {}
                    for nr, nc in neigh_coords:
                        neighbor = self.grid[nr][nc]
                        if neighbor is not None:
                            pop = neighbor["pop"]
                            pop_neighbors.setdefault(pop, []).append(neighbor)
                    colonizer_candidates = []
                    for pop, cells in pop_neighbors.items():
                        # Sum of selective values for neighbors of this population
                        sum_v = sum(cell["v"] for cell in cells)
                        # Compute colonization probability: clamp between 0 and 1.
                        prob = max(0, min(1, 1 - 1/sum_v)) if sum_v > 0 else 0
                        if random.random() < prob:
                            colonizer_candidates.append(pop)
                    if colonizer_candidates:
                        chosen_pop = random.choice(colonizer_candidates)
                        # New cell is created with a random v from that population's distribution.
                        # This code uses the population's mean_v/std_v for newly colonized cells too.
                        mean_v = self.populations[chosen_pop]["mean_v"]
                        std_v  = self.populations[chosen_pop]["std_v"]
                        sampled_v = random.gauss(mean_v, std_v)
                        if sampled_v <= 0:
                            sampled_v = 0.01
                        
                        p = self.populations[chosen_pop]["p"]
                        new_cell = {"pop": chosen_pop, "v": sampled_v, "p": p}
                        new_cells.append((r, c, new_cell))
        # Apply new colonizations
        for r, c, cell in new_cells:
            self.grid[r][c] = cell

    def conflict_phase(self):
        """
        For every occupied cell (defender), if it is adjacent to at least two cells 
        from some enemy population, then each enemy cell (attacker) has a chance 
        to attack.
        """
        bonus_total = 0.5  # Total bonus to share if the defender dies.
        # Loop over each cell as potential defender.
        for r in range(self.rows):
            for c in range(self.cols):
                defender = self.grid[r][c]
                if defender is None:
                    continue
                # Group enemy neighbors by population (different from defender's pop)
                enemy_groups = {}
                for nr, nc in self.get_neighbors(r, c):
                    neighbor = self.grid[nr][nc]
                    if neighbor is not None and neighbor["pop"] != defender["pop"]:
                        enemy_groups.setdefault(neighbor["pop"], []).append((nr, nc, neighbor))
                # For each enemy population with at least 2 neighbors, process conflict.
                for enemy_pop, attackers in enemy_groups.items():
                    if len(attackers) < 2:
                        continue
                    # Use the enemy population's altruism probability for attacks.
                    p_attack = self.populations[enemy_pop]["p"]
                    random.shuffle(attackers)
                    for nr, nc, attacker in attackers:
                        # Only proceed if both attacker and defender still exist.
                        if self.grid[r][c] is None or self.grid[nr][nc] is None:
                            break
                        if random.random() < p_attack:
                            # Simulate the duel: both lose selective value equal to the other's current v.
                            attacker_v = self.grid[nr][nc]["v"]
                            defender_v = self.grid[r][c]["v"]
                            new_attacker_v = attacker_v - defender_v
                            new_defender_v = defender_v - attacker_v
                            self.grid[nr][nc]["v"] = new_attacker_v
                            self.grid[r][c]["v"] = new_defender_v
                            # If attacker’s v falls to zero or below, it dies.
                            if new_attacker_v <= 0:
                                self.grid[nr][nc] = None
                            # If defender’s v falls to zero or below, it dies:
                            if new_defender_v <= 0:
                                self.grid[r][c] = None
                                # Bonus: all neighbors of the defender that belong to enemy_pop share bonus_total.
                                bonus_neighbors = []
                                for ar, ac in self.get_neighbors(r, c):
                                    n2 = self.grid[ar][ac]
                                    if n2 is not None and n2["pop"] == enemy_pop:
                                        bonus_neighbors.append((ar, ac))
                                if bonus_neighbors:
                                    bonus_each = bonus_total / len(bonus_neighbors)
                                    for ar, ac in bonus_neighbors:
                                        if self.grid[ar][ac] is not None:
                                            self.grid[ar][ac]["v"] += bonus_each
                                break  # Stop processing attacks on this defender.
    
    def step(self):
        """One full simulation iteration: colonization then conflict."""
        self.colonization_phase()
        self.conflict_phase()

    def run(self, iterations):
        """Run the simulation for a given number of iterations."""
        for _ in range(iterations):
            self.step()

    def print_grid(self):
        """
        Prints a text representation of the grid.
        Empty cells are shown as '.', and occupied cells display
        the first letter of the population and the current selective value (v).
        """
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell is None:
                    line += ".\t"
                else:
                    line += f"{cell['pop'][0]}({cell['v']:.1f})\t"
            print(line)
        print()


# --------------------------
# Random initialization helper
# --------------------------
def random_init(sim, populations, density=0.1):
    """
    Randomly initializes the simulation grid with a given density.
    Each occupied cell is assigned a random population and a selective value v
    drawn from that population's Gaussian parameters (mean_v, std_v).
    
    :param sim: A HexSimulation instance (with sim.rows x sim.cols grid).
    :param populations: Dict of population parameters, e.g.
                        {
                            "red": {"p": 0.8, "mean_v":1.0, "std_v":0.2},
                            "blue": ...
                        }
    :param density: Fraction of cells to occupy initially (0 <= density <= 1).
    """
    pop_keys = list(populations.keys())
    for r in range(sim.rows):
        for c in range(sim.cols):
            # With probability = density, occupy this cell
            if random.random() < density:
                chosen_pop = random.choice(pop_keys)
                mean_v = populations[chosen_pop]["mean_v"]
                std_v  = populations[chosen_pop]["std_v"]
                sampled_v = random.gauss(mean_v, std_v)
                # Clamp negative or zero values to a small positive
                if sampled_v <= 0:
                    sampled_v = 0.01
                sim.grid[r][c] = {
                    "pop": chosen_pop,
                    "v": sampled_v,
                    "p": populations[chosen_pop]["p"]
                }
            else:
                sim.grid[r][c] = None


# --------------------------
# Visualization code
# --------------------------
def hex_center_odd_r(r, c, s):
    """
    Compute the center coordinates for a pointy-topped hex in an odd-r horizontal layout.
    The spacing formula is:
      x = c*(sqrt(3)*s) + ((sqrt(3)/2)*s if r is odd else 0)
      y = r*(1.5*s)
    """
    x = c * (math.sqrt(3) * s) + ((math.sqrt(3)/2) * s if r % 2 == 1 else 0)
    y = r * (1.5 * s)
    return (x, y)

def run_visual_simulation(sim, hex_size=1.0, interval=500, iterations=100):
    """
    Runs the simulation and displays an animated hexagonal grid.
    
    :param sim: instance of HexSimulation
    :param hex_size: side length of each hexagon (pointy-topped).
    :param interval: delay in milliseconds between animation frames
    :param iterations: number of simulation steps/frames
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    patches_list = []  # will hold tuples (row, col, hexagon patch)
    rows, cols = sim.rows, sim.cols

    # Create hexagon patches for every cell in the grid.
    for r in range(rows):
        for c in range(cols):
            center = hex_center_odd_r(r, c, hex_size)
            # orientation=math.radians(30) => pointy-topped hex with a top vertex pointing upward
            hex_patch = patches.RegularPolygon(
                center, numVertices=6, radius=hex_size,
                orientation=math.radians(0),
                edgecolor='k', facecolor="white"
            )
            ax.add_patch(hex_patch)
            patches_list.append((r, c, hex_patch))
    
    # Set axis limits based on cell centers.
    all_centers = [hex_center_odd_r(r, c, hex_size) for r in range(rows) for c in range(cols)]
    xs = [pt[0] for pt in all_centers]
    ys = [pt[1] for pt in all_centers]
    ax.set_xlim(min(xs) - hex_size, max(xs) + hex_size)
    ax.set_ylim(min(ys) - hex_size, max(ys) + hex_size)
    ax.axis('off')

    current_iteration = 0

    running = False  # State to track if the simulation is running automatically

    def update(frame):
        # Run one simulation step.
        sim.step()
        # Update colors of each hexagon based on current grid state.
        for (r, c, patch) in patches_list:
            cell = sim.grid[r][c]
            if cell is None:
                patch.set_facecolor("white")
            else:
                patch.set_facecolor(cell["pop"])
        ax.set_title(f"Iteration {frame}")
        plt.draw()

    def run_simulation(event):
        nonlocal current_iteration, running
        running = True
        while current_iteration < iterations and running:
            current_iteration += 1
            update(current_iteration)
            plt.pause(interval*0.001)  # Control the speed of the automatic progression

    def next_step(event):
        nonlocal current_iteration
        if current_iteration < iterations:
            current_iteration += 1
            update(current_iteration)

    def pause_simulation(event):
        nonlocal running
        running = False

    # Add buttons for controlling the simulation
    button_width = 0.15

    ax_button_run = plt.axes([0.4, 0.01, button_width, 0.075])  # Run button position
    button_run = Button(ax_button_run, 'Run')
    button_run.on_clicked(run_simulation)

    ax_button_next = plt.axes([0.6, 0.01, button_width, 0.075])  # Next Step button position
    button_next = Button(ax_button_next, 'Next Step')
    button_next.on_clicked(next_step)

    ax_button_pause = plt.axes([0.8, 0.01, button_width, 0.075])  # Pause button position
    button_pause = Button(ax_button_pause, 'Pause')
    button_pause.on_clicked(pause_simulation)

    plt.show()


# --------------------------
# Main block: setup simulation and run visualization
# --------------------------


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    # Define populations with altruism p, plus normal-distribution parameters (mean_v, std_v).
    populations = {
        "red": {"p": 0.8, "mean_v": 1.0, "std_v": 0.3},
        "blue": {"p": 0.3, "mean_v": 1.3, "std_v": 0.3},
        "green": {"p": 0.5, "mean_v": 1.1, "std_v": 0.3}
    }

    rows, cols = 50, 50
    sim = HexSimulation(rows, cols, populations)

    # Randomly initialize 10% of the cells with random populations.
    random_init(sim, populations, density=0.05)

    print("Initial grid:")
    sim.print_grid()

    # Run the animated visualization for 100 steps.
    run_visual_simulation(sim, hex_size=4.0, interval=100, iterations=1000)

