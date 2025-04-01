import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.colors as mcolors
from cuda_square_simulation import SquareSimulation, random_initial_grid, random_action_grid

# Set KMP_DUPLICATE_LIB_OK to avoid errors with MKL and PyTorch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# --------------------------
# Helper function to extract batch 0 labels
# --------------------------
def get_batch0_labels(simulation):
    """
    Extracts the population label for each cell in batch 0.
    
    The simulation grid is assumed to be a torch tensor of shape:
      (nb_batches, number_of_populations, rows, cols)
    For each cell in batch 0, we take the argmax over the population channels.
    If a cell is unoccupied (all channels are 0), we assign it a label of -1.
    
    :param simulation: Simulation instance with attributes:
                       - grid: the torch tensor
                       - pop_ids: dict mapping population names to channel indices.
    :return: A NumPy array of shape (rows, cols) with integer labels.
    """
    grid0 = simulation.grid[0].cpu().numpy()  # shape: (num_populations, rows, cols)
    # For each cell, select the channel with the maximum value.
    labels = grid0.argmax(axis=0)
    # Identify empty cells (where the sum over channels is 0) and mark them as -1.
    empty_mask = grid0.sum(axis=0) == 0
    labels[empty_mask] = -1
    return labels

# --------------------------
# Visualization function for the grid simulation
# --------------------------
def run_visual_simulation_grid(simulation, interval=500, iterations=100, population_colors=None):
    """
    Runs the simulation and displays an animated square grid for batch 0.
    
    The grid is visualized using a discrete colormap. Each cell is colored based
    on the population it belongs to. Unoccupied cells (if any) are shown in white.
    
    :param simulation: Simulation instance with attributes:
                       - grid: torch tensor (nb_batches, number_of_populations, rows, cols)
                       - pop_ids: dict mapping population names to channel indices.
    :param interval: Delay in milliseconds between animation frames.
    :param iterations: Total number of simulation steps/frames.
    :param population_colors: List of color names corresponding to the populations.
           If None, defaults to ['red', 'blue', 'green'].
    """
    if population_colors is None:
        # Default colors; ensure that the order corresponds to the channel indices in simulation.pop_ids.
        population_colors = ['red', 'blue', 'green']
        
    # Create a colormap for the populations.
    # Note: The imshow will display integer labels 0, 1, 2,... using these colors.
    cmap = mcolors.ListedColormap(population_colors)
    
    # Set up the figure and axis.
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Get the initial labels for batch 0.
    labels = get_batch0_labels(simulation)  # shape (rows, cols)
    im = ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(population_colors)-0.5)
    ax.axis('off')
    
    current_iteration = 0
    running = False

    def update_grid():
        nonlocal current_iteration
        new_labels = get_batch0_labels(simulation)
        im.set_data(new_labels)
        ax.set_title(f"Iteration {current_iteration}")
        plt.draw()

    def update(frame):
        action_grid = random_action_grid(simulation.batch_size, simulation.rows, simulation.cols, simulation.device)

        simulation.step(action_grid)  # Run one simulation step (assumes simulation.step() updates simulation.grid)

        nonlocal current_iteration
        current_iteration += 1
        update_grid()

    def run_simulation(event):
        nonlocal running, current_iteration
        running = True
        while current_iteration < iterations and running:
            update(current_iteration)
            plt.pause(interval * 0.001)  # Convert milliseconds to seconds

    def next_step(event):
        nonlocal current_iteration
        if current_iteration < iterations:
            update(current_iteration)

    def pause_simulation(event):
        nonlocal running
        running = False

    # Add buttons for controlling the simulation.
    button_width = 0.15

    ax_button_run = plt.axes([0.4, 0.01, button_width, 0.075])
    button_run = Button(ax_button_run, 'Run')
    button_run.on_clicked(run_simulation)

    ax_button_next = plt.axes([0.6, 0.01, button_width, 0.075])
    button_next = Button(ax_button_next, 'Next Step')
    button_next.on_clicked(next_step)

    ax_button_pause = plt.axes([0.8, 0.01, button_width, 0.075])
    button_pause = Button(ax_button_pause, 'Pause')
    button_pause.on_clicked(pause_simulation)

    plt.show()


# --------------------------
# Example usage:
# --------------------------
if __name__ == "__main__":
    # For this example, we assume that:
    # - A SquareSimulation class exists that implements simulation.step() and holds the grid.
    # - The simulation is already initialized via the vectorized random_initial_grid.
    
    # Example: define parameters and create the simulation.
    nb_batches = 100
    rows, cols = 50, 50
    populations = {
        "red": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
        "blue": {"p": 0.3, "mean_v": 1.0, "std_v": 0.2},
        "green": {"p": 0.3, "mean_v": 1.0, "std_v": 0.2},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # The SquareSimulation class is assumed to have attributes:
    #   - grid (a torch tensor)
    #   - pop_ids: e.g., {"red": 0, "blue": 1, "green": 2}
    # and a method step() that updates grid.
    simulation = SquareSimulation(nb_batch=nb_batches, rows=rows, cols=cols,
                                  populations=populations, device=device)
    
    # Initialize the simulation grid (using your vectorized function).
    random_initial_grid(simulation, populations, nb_batches, rows, cols, device)
    
    # Run the visual simulation for batch 0.
    run_visual_simulation_grid(simulation, interval=100, iterations=1000)
