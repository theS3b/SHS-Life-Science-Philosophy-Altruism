import random
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button
from hex_simulation import HexSimulation

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

