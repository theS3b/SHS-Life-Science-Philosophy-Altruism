import random
import numpy as np
import torch
import torch.nn as nn
import time

# Dummy profile decorator for when not using kernprof
try:
    profile
except NameError:
    def profile(func):
        return func

class SquareSimulation:
    FITNESS_DONATION = 0.1  # Percentage of fitness donated to the neighbor
    EPS = 1e-6

    def __init__(self, nb_batch, rows, cols, populations, device, initial_grid=None):
        """
        :param rows: number of rows in the grid.
        :param cols: number of columns in the grid.
        :param populations: dict mapping population id to parameters, e.g.
                            {"red": {"p": 0.8, "mean_v": 1.0, "std_v": 0.2}, ... }
        :param initial_grid: optional initial grid (3D list). If None, grid is empty.
        """
        self.rows = rows
        self.cols = cols
        self.batch_size = nb_batch
        
        # Associate an id to each population
        self.populations = populations  # population parameters
        self.pop_ids = list(populations.keys())
        self.pop_ids.sort()  # Sort population ids for consistent ordering
        self.pop_ids = {pop_id: i for i, pop_id in enumerate(self.pop_ids)}
        self.ids_to_pop = {i: pop_id for i, pop_id in enumerate(self.pop_ids)}
        self.number_of_populations = len(self.pop_ids)

        if initial_grid is not None:
            self.grid = initial_grid.to(device)
        else:
            # Grid contains the fitness for each cell
            self.grid = torch.zeros((nb_batch, self.number_of_populations, rows, cols), dtype=torch.float32).to(device) # Initialize with 0 for empty cells

        self.device = device
        self.CIRC_KERNEL = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=self.grid.dtype, device=self.device).view(1, 1, 3, 3)

    
    def colonization_phase(self):
        """Each empty cell can be colonized by a neighboring population."""

        # Get empty cells for each (batch, row, col) it must be -1 to be empty
        empty_cells = (self.grid == 0).all(dim=1).unsqueeze(1)  # shape: [batch, 1, rows, cols]

        # Define the kernel (same for all channels)
        kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=self.device)
        kernel = kernel / kernel.sum()  # Normalize kernel

        # Duplicate kernel for each channel: [out_channels=3, in_channels/groups=1, kH, kW]
        kernel = kernel.view(1, 1, 3, 3).repeat(self.number_of_populations, 1, 1, 1)  # shape: [3,1,3,3]

        # Pad input
        grid_padded = nn.CircularPad2d(1)(self.grid)  # Pad with reflection to avoid border issues (suppose infinity grid)

        # Apply depthwise convolution: groups = number of channels
        new_fitness = torch.nn.functional.conv2d(grid_padded, kernel, stride=1, padding=0, groups=self.number_of_populations)

        # Set fitness to 0 where the cell is not empty
        empty_cells_expanded = empty_cells.expand(self.batch_size, self.number_of_populations, self.rows, self.cols)  # Expand to match the number of populations
        new_fitness[~empty_cells_expanded] = 0

        # Keep only the maximum fitness value for each batch and cell
        # Step 1: Find the maximum value and its index for each batch
        _, max_indices = torch.max(new_fitness, dim=1, keepdim=True)

        # Step 2: Create a mask that is True where the max occurs
        mask = torch.arange(new_fitness.size(1), device=new_fitness.device).view(1, -1, 1, 1) == max_indices

        # Step 3: Zero out everything that's not the max
        new_fitnesses_only = new_fitness * mask.float()

        # Update the grid with the new fitness values
        self.grid = self.grid + new_fitnesses_only  # Add the new fitness values to the grid

    @profile
    def conflict_phase(self, action_grid):
        # If action = 0, do nothing
        # If action = 1-8, give fitness to the cell in the direction of the action
        # If action = 9-16, attack the opponent cell
        flat_grid, _ = torch.max(self.grid, dim=1, keepdim=True)  # shape: (batch, 1, n, m)

        # Donnations
        self.manage_donations(flat_grid, action_grid)

        # Attacks
        self.manage_attacks(flat_grid, action_grid)

    @profile
    def manage_donations(self, flat_grid, action_grid):

        shifted_action = action_grid - 1
        up = torch.roll(torch.where(shifted_action == 0, flat_grid, 0), shifts=-1, dims=2)
        up_right = torch.roll(torch.where(shifted_action == 1, flat_grid, 0), shifts=(-1, 1), dims=(2, 3))
        right = torch.roll(torch.where(shifted_action == 2, flat_grid, 0), shifts=1, dims=3)
        down_right = torch.roll(torch.where(shifted_action == 3, flat_grid, 0), shifts=(1, 1), dims=(2, 3))
        down = torch.roll(torch.where(shifted_action == 4, flat_grid, 0), shifts=1, dims=2)
        down_left = torch.roll(torch.where(shifted_action == 5, flat_grid, 0), shifts=(1, -1), dims=(2, 3))
        left = torch.roll(torch.where(shifted_action == 6, flat_grid, 0), shifts=-1, dims=3)
        up_left = torch.roll(torch.where(shifted_action == 7, flat_grid, 0), shifts=(-1, -1), dims=(2, 3))

        # Combine all contributions
        contributions = up + up_right + right + down_right + down + down_left + left + up_left
        contributions = contributions * self.FITNESS_DONATION  # Scale by donation percentage

        # Remap to original dimensions
        new_grid = self.grid + (self.grid > SquareSimulation.EPS) * contributions  # Add contributions only to non-empty cells

        # Decrease by 10% for the cells that gave fitness
        new_grid = torch.where((shifted_action >= 0) & (shifted_action <= 7), new_grid * (1 - self.FITNESS_DONATION), new_grid)

        new_grid = torch.where(new_grid < 0, 0, new_grid)  # Ensure no negative values

        self.grid = new_grid  # Update the grid with the new fitness values

    @profile
    def manage_attacks(self, flat_grid, action_grid):
        # Get the directional index (0 to 7) for attack actions:
        shifted_action = (action_grid - 9)

        empty_cells = (flat_grid <= 1e-6).all(dim=1)  # shape: [batch, rows, cols]

        nb_existing_neighbors = torch.nn.functional.conv2d(
            nn.CircularPad2d(1)((~empty_cells).float()).unsqueeze(1),
            self.CIRC_KERNEL
        )

        # Define the 8 directional shifts (row, col) corresponding to neighbors:
        # 0: up, 1: up-right, 2: right, 3: down-right,
        # 4: down, 5: down-left, 6: left, 7: up-left.
        shifts = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                (1, 0), (1, -1), (0, -1), (-1, -1)]

        # Process each direction separately
        iterations = enumerate(shifts)

        # Avoid bias of attacking in one direction first
        random.shuffle(list(iterations))
        
        for i, (shift_row, shift_col) in iterations:
            # Create mask for cells whose shifted_action equals this direction (attack events)
            mask = (shifted_action == i)
            if mask.sum() == 0:
                continue  # No attack events in this direction
            
            # Get attacker values from flat_grid where attack occurs.
            attacker_vals = torch.where(mask, flat_grid, torch.zeros_like(flat_grid))

            # Get defender values from the neighbor cell by rolling the flat_grid.
            defender_mapped_to_attacker_vals = torch.roll(flat_grid, shifts=(-shift_row, -shift_col), dims=(2, 3))
            defender_mapped_to_attacker_vals = torch.where(mask, defender_mapped_to_attacker_vals, torch.zeros_like(flat_grid))
            
            # Compute the value to subtract: the minimum of the two values.
            vals = torch.min(attacker_vals, defender_mapped_to_attacker_vals)
            
            # Subtract val from the attacker cell.
            flat_grid = torch.where(mask, flat_grid - vals, flat_grid)

            # For the defender, roll the mask to align with neighbor positions.
            defender_mask = torch.roll(mask, shifts=(shift_row, shift_col), dims=(2, 3))
            vals_mapped_to_defender = torch.roll(vals, shifts=(shift_row, shift_col), dims=(2, 3))
            flat_grid = torch.where(defender_mask, flat_grid - vals_mapped_to_defender, flat_grid)

            attacker_dying_cells = (flat_grid <= 1e-6) & mask
            defender_dying_cells = (flat_grid <= 1e-6) & defender_mask

            bonus = torch.zeros_like(flat_grid)  # Reset bonus for each direction
            bonus[attacker_dying_cells] = vals[attacker_dying_cells]  # Assign bonus to dying cells
            bonus[defender_dying_cells] = vals_mapped_to_defender[defender_dying_cells]  # Assign bonus to dying cells

            bonus = bonus / nb_existing_neighbors.clamp(min=1e-6)  # Avoid division by zero

            # Distribute bonus using convolution with circular padding.
            distributed_bonus = torch.nn.functional.conv2d(
                torch.nn.functional.pad(bonus, (1, 1, 1, 1), mode='circular'),
                self.CIRC_KERNEL
            )

            # Remove bonus from empty cells
            distributed_bonus[empty_cells.unsqueeze(1)] = 0
            flat_grid += distributed_bonus  # Update flat_grid with distributed bonus

            # Enusre no negative values
            flat_grid = torch.where(flat_grid < 0, 0, flat_grid)

        # Update grid
        self.grid = (self.grid > SquareSimulation.EPS) * flat_grid


    @profile
    def step(self, action_grid=None):
        """One full simulation iteration: colonization then conflict."""
        self.colonization_phase()

        if action_grid is not None:
            self.conflict_phase(action_grid)

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
                cell = self.grid[0, :, r, c]
                if cell.sum() == 0:
                    line += "."  # Empty cell
                else:
                    # Find the population with the highest fitness value
                    max_pop_id = torch.argmax(cell).item()
                    line += f"{self.ids_to_pop[max_pop_id][0]}({cell[max_pop_id].item():.2f}) " # Display population id and fitness value

            print(line)
            print()


def random_initial_grid(simulation, populations, nb_batches, rows, cols, device):
    # Vectorized grid initialization
    with torch.no_grad():
        probs = {
            "red": populations["red"]["p"],
            "blue": populations["blue"]["p"],
            "green": populations["green"]["p"],
        }
        means = {
            "red": populations["red"]["mean_v"],
            "blue": populations["blue"]["mean_v"],
            "green": populations["green"]["mean_v"],
        }
        stds = {
            "red": populations["red"]["std_v"],
            "blue": populations["blue"]["std_v"],
            "green": populations["green"]["std_v"],
        }

        # Generate a uniform random grid for all batches
        rand_vals = torch.rand((nb_batches, rows, cols), device=device)

        # Allocate empty grid
        simulation.grid = torch.zeros((nb_batches, simulation.number_of_populations, rows, cols), dtype=torch.float32, device=device)

        thresholds = torch.tensor([
            probs["red"],
            probs["red"] + probs["blue"]
        ], device=device).view(1, 2, 1, 1)

        # Classify cells
        is_red = rand_vals < thresholds[:, 0, :, :]
        is_blue = (rand_vals >= thresholds[:, 0, :, :]) & (rand_vals < thresholds[:, 1, :, :])
        is_green = rand_vals >= thresholds[:, 1, :, :]

        # Sample values for each population
        red_vals = torch.normal(means["red"], stds["red"], size=(nb_batches, rows, cols), device=device)
        blue_vals = torch.normal(means["blue"], stds["blue"], size=(nb_batches, rows, cols), device=device)
        green_vals = torch.normal(means["green"], stds["green"], size=(nb_batches, rows, cols), device=device)

        simulation.grid[:, simulation.pop_ids["red"]] = red_vals * is_red
        simulation.grid[:, simulation.pop_ids["blue"]] = blue_vals * is_blue
        simulation.grid[:, simulation.pop_ids["green"]] = green_vals * is_green


# Example usage:
def main():
    print("CUDA available:", torch.cuda.is_available())

    nb_batches = 10000
    rows, cols = 50, 50
    populations = {
        "red": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
        "blue": {"p": 0.3, "mean_v": 1.0, "std_v": 0.2},
        "green": {"p": 0.3, "mean_v": 1.0, "std_v": 0.2},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulation = SquareSimulation(nb_batch=nb_batches, rows=rows, cols=cols, populations=populations, device=device)

    # Initialize the grid with random populations
    random_initial_grid(simulation, populations, nb_batches, rows, cols, device)

    print("Start")

    # Run the simulation for a few steps
    time_now = time.time()
    for _ in range(50):
        action_grid = torch.randint(0, 17, (1, rows, cols), device=device)  # Random actions
        simulation.step(action_grid)

    time_elapsed = time.time() - time_now
    print(f"Time elapsed: {time_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()