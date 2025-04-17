import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.distributions.multivariate_normal import MultivariateNormal

# Dummy profile decorator for when not using kernprof
try:
    profile
except NameError:
    def profile(func):
        return func

class SquareSimulation:
    FITNESS_DONATION = 0.1  # Percentage of fitness donated to the neighbor
    FITNESS_DONATION_BONUS = 0.1
    EPS = 1e-6
    COLONIZE_PROB_ONE = 2 # 2 fitness for prob 100% of colonization
    REWARD_FOR_DONE = 1.0  # Reward for reaching the done condition
    FITNESS_GROWTH_VALUE = 0.0 # number of fitness points gained at every step

    def __init__(self, nb_batch, rows, cols, populations, device, observation_size=5, done_population=0.9, initial_grid=None):
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
        self.observation_size = observation_size
        self.done_population = done_population  # Population density threshold for done condition

    
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

        uniform_rand = SquareSimulation.COLONIZE_PROB_ONE * torch.rand_like(new_fitnesses_only)  # shape: (batch, 1, rows, cols)
        new_fitnesses_only = torch.where(uniform_rand < new_fitnesses_only, new_fitnesses_only, torch.zeros_like(new_fitnesses_only))

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
        shifted_action = shifted_action.unsqueeze(1)

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
        contributions = contributions * (self.FITNESS_DONATION + self.FITNESS_DONATION_BONUS)  # Scale by donation percentage

        # WARNING! If you donate to empty cell, you just loose fitness so this is not really a zero sum game
        # Remap to original dimensions
        contributions_mapped = (self.grid > SquareSimulation.EPS) * contributions  # Add contributions only to non-empty cells
        new_grid = self.grid + contributions_mapped

        # Decrease by 10% for the cells that gave fitness
        new_grid = torch.where((shifted_action >= 0) & (shifted_action <= 7), new_grid * (1 - self.FITNESS_DONATION), new_grid)

        new_grid = torch.where(new_grid < 0, 0, new_grid)  # Ensure no negative values

        self.grid = new_grid  # Update the grid with the new fitness values

    @profile
    def manage_attacks(self, flat_grid, action_grid):
        # Get the directional index (0 to 7) for attack actions:
        shifted_action = (action_grid - 9)

        empty_cells = (flat_grid <= SquareSimulation.EPS).all(dim=1)  # shape: [batch, rows, cols]

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
        iterations = list(enumerate(shifts))

        # Avoid bias of attacking in one direction first
        random.shuffle(iterations)
        
        for i, (shift_row, shift_col) in iterations:
            # Create mask for cells whose shifted_action equals this direction (attack events)
            mask = (shifted_action == i).unsqueeze(1)  # shape: [batch, 1, rows, cols]
            
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

            attacker_dying_cells = (flat_grid <= SquareSimulation.EPS) & mask
            defender_dying_cells = (flat_grid <= SquareSimulation.EPS) & defender_mask

            bonus = torch.zeros_like(flat_grid)  # Reset bonus for each direction
            bonus[attacker_dying_cells] = vals[attacker_dying_cells]  # Assign bonus to dying cells
            bonus[defender_dying_cells] = vals_mapped_to_defender[defender_dying_cells]  # Assign bonus to dying cells

            bonus = bonus / nb_existing_neighbors.clamp(min=SquareSimulation.EPS)  # Avoid division by zero

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
    def fitness_growth(self):
        # Augment the fitness of every populated cells by a fix number : growth_value
        mask = self.grid > self.EPS
        self.grid += mask*self.FITNESS_GROWTH_VALUE


    @profile
    def step(self, action_grid=None):
        """One full simulation iteration: colonization then conflict."""
        past_grid = self.grid.clone()

        self.colonization_phase()

        if action_grid is not None:
            self.conflict_phase(action_grid)

        self.fitness_growth()

        # Compute rewards
        rewards, done_batch, info = self.compute_rewards(past_grid)

        return rewards, done_batch, info

    def compute_rewards(self, past_grid):
        """Compute rewards based on the change in fitness values.
        
        returns:
            rewards: tensor of shape (nb_pop,)
            done_batch: tensor of shape (batch_size,)
            info: dict with additional information
        """

        rewards = self.grid - past_grid
        rewards = rewards.sum(dim=(0, 2, 3))
        rewards = rewards / (self.rows * self.cols)  # Normalize by the number of cells

        # Define done if one population populates x% of the grid
        done_batch_per_pop = torch.sum(self.grid > SquareSimulation.EPS, dim=(2, 3)) / (self.rows * self.cols) > self.done_population
        done_batch = done_batch_per_pop.any(dim=1)  # Check if any population has reached the threshold in each batch

        # TODO We could add bigger reward for the population that reached the threshold
        # rewards += done_batch_per_pop.sum(dim=0) * self.REWARD_FOR_DONE

        info = {
            # TODO @Romain: Add more info for logging
        }

        # Reset done batches only
        self.reset(done_batch.nonzero(as_tuple=True)[0])

        return rewards, done_batch, info

    
    def reset(self, batch_ids = None):
        """Reset the grid for the specified batch ids."""
        if batch_ids is None:
            batch_ids = torch.arange(self.batch_size, device=self.device)

        nb_batches = len(batch_ids)

        # Generate a uniform random grid for all batches
        rand_vals = torch.rand((nb_batches, self.rows, self.cols), device=self.device)

        # Allocate empty grid
        grid = torch.zeros((nb_batches, self.number_of_populations, self.rows, self.cols), dtype=torch.float32, device=self.device)

        probs = torch.tensor([self.populations[pop_id]["p"] for pop_id in self.pop_ids], device=self.device)

        thresholds = torch.cumsum(probs, dim=0).view(1, -1, 1, 1)  # shape: (1, x, 1, 1)
        
        # Classify cells
        for i, pop_id in enumerate(self.pop_ids):
            if i == 0:
                is_pop = rand_vals < thresholds[:, i, :, :]
            else:
                is_pop = (rand_vals >= thresholds[:, i-1, :, :]) & (rand_vals < thresholds[:, i, :, :])

            # Sample values for each population
            mean_v = self.populations[pop_id]["mean_v"]
            std_v = self.populations[pop_id]["std_v"]
            pop_vals = torch.normal(mean_v, std_v, size=(nb_batches, self.rows, self.cols), device=self.device)

            grid[:, self.pop_ids[pop_id]] = pop_vals * is_pop

        self.grid[batch_ids] = grid  # Update the grid for the specified batch ids


    def run(self, iterations):
        """Run the simulation for a given number of iterations."""
        for _ in range(iterations):
            self.step()

    def get_deep_observations(self, population_id):
        B, C, H, W = self.grid.shape
        obs_size = self.observation_size

        # Get indices of interest
        indices = torch.nonzero(self.grid[:, population_id, ...] > self.EPS, as_tuple=False)  # shape: (N, 3)
        batch_idx, y_idx, x_idx = indices[:, 0], indices[:, 1], indices[:, 2]

        # Pad the spatial dims (H, W) with 2 pixels on each side
        padded = nn.CircularPad2d(obs_size // 2)(self.grid)  # shape: (B, C, H + obs_size - 1, W + obs_size - 1)

        # Use unfold to get all 5x5 patches from each image in batch (suppose obs_size = 5)
        # unfold returns shape (B, 25, (H * W))
        patches = F.unfold(padded, kernel_size=obs_size)  # shape: (B, C * obs_size * obs_size, H * W)

        patches_ = patches.view(B, C, obs_size*obs_size, H, W)  # shape: (B, C, 25, H, W)

        # Get the patches corresponding to the indices of interest
        patches_of_interest = patches_[batch_idx, :, :, y_idx, x_idx]  # shape: (N, C, 25)
        
        # Reshape to (N, C, obs_size, obs_size)
        patches_of_interest = patches_of_interest.view(-1, C, obs_size, obs_size)  # shape: (N, C, 5, 5)

        return patches_of_interest, indices  # Return patches and their indices


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

    def get_random_action_grid(self):
        """Generates a random action grid for the simulation."""
        # Generate random actions (0-16) for each cell in the grid
        action_grid = torch.zeros((self.batch_size, self.rows, self.cols), dtype=torch.float32, device=self.device)

        # Small percentage are mapped to 1-8 (donations)
        random_choice = torch.randint(1, 100, (self.batch_size, self.rows, self.cols), device=self.device).float()
        
        action_grid = torch.where(random_choice > 16, 0, random_choice)  # 0: do nothing
        
        # Remove actions on empty cells
        empty_cells = (~(self.grid > self.EPS)).all(dim=1)  # shape: (batch, rows, cols)

        action_grid[empty_cells] = 0

        return action_grid

    def inteligent_agent_action_grid(self):
        """Generates an action grid where each population does not attack allies or donate to enemies."""

        EPS = 1e-6

        # Create a tensor of every possible action
        n_actions = 17

        # Generate gaussian tensor, the idea is that argmax will be random, we can now manipulate the tab to prevent agent to do unproductive actions.

        # [nb_batch, rows, cols, n_actions]
        allowed_actions = torch.normal(mean=1.0, std=0.1, size=(self.batch_size, self.rows, self.cols, n_actions), dtype=torch.float32, device=self.device).clamp(0)

        # exemple : the line below prevent the agent to attack an other one
        #allowed_actions[:,:,:, 9:] = -1

        for pop_id in range(self.number_of_populations):

            # Every possible action id:
                # 0: do nothing

                # donnation in direction :
                # 1: up, 2: up-right, 3: right, 4: down-right,
                # 5: down, 6: down-left, 7: left, 8: up-left.

                # attack in direction :
                # 9: up, 10: up-right, 11: right, 12: down-right,
                # 13: down, 14: down-left, 15: left, 16: up-left.

            ally_mask = self.grid[:,pop_id,...] > EPS # [batch, rows, cols]

            ally_up = torch.roll(ally_mask, shifts=1, dims=1) > EPS #up 9
            ally_up_right = torch.roll(ally_mask, shifts=(1, -1), dims=(1, 2)) > EPS #up_right 10
            ally_right = torch.roll(ally_mask, shifts=-1, dims=2) > EPS #right 11
            ally_down_right = torch.roll(ally_mask, shifts=(-1, -1), dims=(1, 2)) > EPS #down_right 12
            ally_down = torch.roll(ally_mask, shifts=-1, dims=1) > EPS # down 13
            ally_down_left = torch.roll(ally_mask, shifts=(-1, 1), dims=(1, 2)) > EPS #down_left 14
            ally_left = torch.roll(ally_mask, shifts=1, dims=2) > EPS #left 15
            ally_up_left = torch.roll(ally_mask, shifts=(1, 1), dims=(1, 2)) > EPS #up_left 16

            ally_neighbors = [ally_up, ally_up_right, ally_right, ally_down_right, ally_down, ally_down_left, ally_left, ally_up_left]

            for idx, n in enumerate(ally_neighbors):
                donnation_action = idx+1
                attack_action = idx+9

                # cannot attack if the neighbors is a ally
                direction_ally_mask  = n & ally_mask
                b, r, c = torch.where(direction_ally_mask)
                allowed_actions[b,r,c, attack_action] = -1

                # donation if neighbors is not a ally donation is forbidden
                b, r, c = torch.where(~n & ally_mask)
                allowed_actions[b,r,c, donnation_action] = -1

        # Create a mask to erase action for unpopulated cells
        population_mask = (self.grid > EPS).any(dim=1)

        return allowed_actions.argmax(-1) * population_mask

    def one_intelligent_population_action_grid(self, population_id):
        """Generates an action grid where one population does not attack allies or donate to enemies."""
        """Every other population is inactive"""
        """The intelligent population is the one having id :population_id"""

        EPS = 1e-6

        # Create a tensor of every possible action
        n_actions = 17

        # Generate gaussian tensor, the idea is that argmax will be random, we can now manipulate the tab to prevent agent to do unproductive actions.

        # [nb_batch, rows, cols, n_actions]
        allowed_actions = torch.normal(mean=1.0, std=0.1, size=(self.batch_size, self.rows, self.cols, n_actions), dtype=torch.float32, device=self.device).clamp(0)

        # exemple : the line below prevent the agent to attack an other one
        #allowed_actions[:,:,:, :9] = -1

        ally_mask = self.grid[:,population_id,...] > EPS # [batch, rows, cols]

        # Every possible action id:
            # 0: do nothing

            # donnation in direction :
            # 1: up, 2: up-right, 3: right, 4: down-right,
            # 5: down, 6: down-left, 7: left, 8: up-left.

            # attack in direction :
            # 9: up, 10: up-right, 11: right, 12: down-right,
            # 13: down, 14: down-left, 15: left, 16: up-left.

        ally_up = torch.roll(ally_mask, shifts=1, dims=1) > EPS #up 9
        ally_up_right = torch.roll(ally_mask, shifts=(1, -1), dims=(1, 2)) > EPS #up_right 10
        ally_right = torch.roll(ally_mask, shifts=-1, dims=2) > EPS #right 11
        ally_down_right = torch.roll(ally_mask, shifts=(-1, -1), dims=(1, 2)) > EPS #down_right 12
        ally_down = torch.roll(ally_mask, shifts=-1, dims=1) > EPS # down 13
        ally_down_left = torch.roll(ally_mask, shifts=(-1, 1), dims=(1, 2)) > EPS #down_left 14
        ally_left = torch.roll(ally_mask, shifts=1, dims=2) > EPS #left 15
        ally_up_left = torch.roll(ally_mask, shifts=(1, 1), dims=(1, 2)) > EPS #up_left 16

        ally_neighbors = [ally_up, ally_up_right, ally_right, ally_down_right, ally_down, ally_down_left, ally_left, ally_up_left]

        for idx, n in enumerate(ally_neighbors):
            donnation_action = idx+1
            attack_action = idx+9

            # cannot attack if the neighbors is a ally
            direction_ally_mask  = n & ally_mask
            b, r, c = torch.where(direction_ally_mask)
            allowed_actions[b,r,c, attack_action] = -1

            # donation if neighbors is not a ally donation is forbidden
            b, r, c = torch.where(~n & ally_mask)
            allowed_actions[b,r,c, donnation_action] = -1

        # Create a mask to cancel other populatin actions
        population_mask = (self.grid[:, population_id] > EPS)

        return allowed_actions.argmax(-1) * population_mask

    def one_intelligent_population_vs_random_action_grid(self, population_id):
        """Generates an action grid where one population does not attack allies or donate to enemies."""
        """Every other population takes uniformly random action"""
        """The intelligent population is the one having id :population_id"""
        EPS = 1e-6

        action_grid = self.get_random_action_grid()

        population_id_action = self.one_intelligent_population_action_grid(population_id)
        population_id_mask = self.grid[:,population_id] > EPS

        action_grid = (action_grid * ~population_id_mask) + population_id_action

        population_mask = (self.grid > EPS).any(dim=1)

        return action_grid * population_mask

    def random_initial_grid(self, populations, nb_batches, rows, cols, device):
        """Vectorized uniformly random grid initialization"""
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
            self.grid = torch.zeros((nb_batches, self.number_of_populations, rows, cols), dtype=torch.float32, device=device)

            thresholds = torch.tensor([
                probs["red"],
                probs["red"] + probs["blue"],
                probs["red"] + probs["blue"] + probs["green"]
            ], device=device).view(1, 3, 1, 1)

            # Classify cells
            is_red = rand_vals < thresholds[:, 0, :, :]
            is_blue = (rand_vals >= thresholds[:, 0, :, :]) & (rand_vals < thresholds[:, 1, :, :])
            is_green = (rand_vals >= thresholds[:, 1, :, :]) & (rand_vals < thresholds[:, 2, :, :])

            # Sample values for each population
            red_vals = torch.normal(means["red"], stds["red"], size=(nb_batches, rows, cols), device=device)
            blue_vals = torch.normal(means["blue"], stds["blue"], size=(nb_batches, rows, cols), device=device)
            green_vals = torch.normal(means["green"], stds["green"], size=(nb_batches, rows, cols), device=device)

            self.grid[:, self.pop_ids["red"]] = red_vals * is_red
            self.grid[:, self.pop_ids["blue"]] = blue_vals * is_blue
            self.grid[:, self.pop_ids["green"]] = green_vals * is_green

            # Keep only the maximum fitness value for each batch and cell
            # Step 1: Find the maximum value and its index for each batch
            _, max_indices = torch.max(self.grid, dim=1, keepdim=True)

            # Step 2: Create a mask that is True where the max occurs
            mask = torch.arange(self.grid.size(1), device=self.grid.device).view(1, -1, 1, 1) == max_indices

            # Step 3: Zero out everything that's not the max
            new_fitnesses_only = self.grid * mask.float()

            # Update the grid with the new fitness values
            self.grid = new_fitnesses_only

            # Ensure no negative values
            self.grid = torch.where(self.grid < 0, 0, self.grid)

    def random_initial_grid_with_gaussians(self):
        """Grid initialization containing one randomly centered 2d gaussian distribution of cells"""
        with torch.no_grad():

            nb_population = self.number_of_populations

            # select variance and max individual in gaussian (in fct of the grid size)
            nb_individuals = self.rows*self.cols
            variance_gaussian = self.rows*self.cols/60

            probas = torch.tensor([self.populations["red"]["p"], self.populations["blue"]["p"], self.populations["green"]["p"]], device=self.device)

            samples_per_pop = (nb_individuals * probas).round().long()

            # Init Grid
            self.grid = torch.zeros((self.batch_size, nb_population, self.rows, self.cols), dtype=torch.float32, device=self.device)

            # Choose random center for gaussians
            centers_y = torch.randint(0, self.rows, (self.batch_size, nb_population), device=self.device)
            centers_x = torch.randint(0, self.cols, (self.batch_size, nb_population), device=self.device)
            centers = torch.stack([centers_y, centers_x], dim=-1)  # [B, P, 2]

            # Create a gaussian centered in (0,0)
            cov = torch.eye(2, device=self.device) * variance_gaussian #TODO check if we want to add covariance
            mvn = MultivariateNormal(loc=torch.zeros(2, device=self.device), covariance_matrix=cov)

            for pop_idx, pop_sample in enumerate(samples_per_pop):
                if pop_sample==0:
                    continue

                # Sample pop_sample random point in gaussians
                samples = mvn.sample((self.batch_size, 1, pop_sample)) # [B, 1, pop_sample, 2]

                # Center the sample around the position
                positions = centers[:, pop_idx].unsqueeze(1).unsqueeze(2) + samples  # (B, P, N, 2)
                positions = positions.round().long()

                # modulo the coordinate for periodic bound
                positions[..., 0] = positions[..., 0] %self.rows
                positions[..., 1] = positions[..., 1] %self.cols

                #b_idx = torch.arange(nb_batches, device=device).view(-1, 1, 1).expand(-1, nb_population, nb_individuals)

                b_idx = torch.arange(self.batch_size, device=self.device).view(-1, 1).expand(-1, pop_sample)  # [B, N]
                p_idx = torch.full_like(b_idx, pop_idx)  # [B, N]

                # Add one to the selected grid
                self.grid[b_idx.reshape(-1),
                                p_idx.reshape(-1),
                                positions[..., 0].reshape(-1),
                                positions[..., 1].reshape(-1)] = 1

            means = {
                "red": self.populations["red"]["mean_v"],
                "blue": self.populations["blue"]["mean_v"],
                "green": self.populations["green"]["mean_v"],
            }
            stds = {
                "red": self.populations["red"]["std_v"],
                "blue": self.populations["blue"]["std_v"],
                "green": self.populations["green"]["std_v"],
            }
            for pop_idx, pop in enumerate(['red', 'blue', 'green']):
                self.grid[:, pop_idx] *= torch.normal(mean=means[pop], std=stds[pop], size=self.grid[:, 0].size(), device=self.grid.device)

            # Keep max fitness per pop to prevent conflicts
            # Step 1: Find the maximum value and its index for each batch
            _, max_indices = torch.max(self.grid, dim=1, keepdim=True)

            # Step 2: Create a mask that is True where the max occurs
            mask = torch.arange(self.grid.size(1), device=self.device).view(1, -1, 1, 1) == max_indices

            # Step 3: Zero out everything that's not the max
            self.grid *= mask.float()

# Example usage:
def main():
    print("CUDA available:", torch.cuda.is_available())

    nb_batches = 10000
    rows, cols = 50, 50
    populations = {
        "red": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
        "blue": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
        "green": {"p": 0.2, "mean_v": 1.0, "std_v": 0.2},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulation = SquareSimulation(nb_batch=nb_batches, rows=rows, cols=cols, populations=populations, device=device)

    # Initialize the grid with random populations
    simulation.random_initial_grid(simulation, populations, nb_batches, rows, cols, device)

    # Run the simulation for a few steps
    time_now = time.time()
    for _ in range(10):
        action_grid = torch.randint(0, 17, (nb_batches, rows, cols), device=device)  # Random actions
        simulation.step(action_grid)

    time_elapsed = time.time() - time_now
    print(f"Time elapsed: {time_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()