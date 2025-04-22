from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class Actions(Enum):
    nothing = 0
    donate_up = 1
    donate_up_right = 2
    donate_right = 3
    donate_down_right = 4
    donate_down = 5
    donate_down_left = 6
    donate_left = 7
    donate_up_left = 8
    attack_up = 9
    attack_up_right = 10
    attack_right = 11
    attack_down_right = 12
    attack_down = 13
    attack_down_left = 14
    attack_left = 15
    attack_up_left = 16

class SquareAdaptedGymSimulation(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "max_fitness": 2.0, "min_fitness": 0.0}

    def __init__(self, efficient_environment, observation_size, clever_pop_id, render_mode=None):
        super().__init__()

        self.underlying_env = efficient_environment

        # Define the observation space.
        # Each observation corresponds to a neighborhood: (3, 5, 5)
        # Adjust low/high according to your fitness value ranges.
        self.observation_space = spaces.Box(
            low=self.metadata["min_fitness"],
            high=self.metadata["max_fitness"],
            shape=(self.underlying_env.number_of_populations, observation_size, observation_size),
            dtype=np.float32,
        )

        # We have the actions Actions
        self.action_space = spaces.Discrete(len(Actions))
        self.clever_pop_id = clever_pop_id

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.underlying_env.reset()

        observation_clever_pop = self.underlying_env.get_deep_observations(self.clever_pop_id).cpu().numpy()
        info = {}
        

        if self.render_mode == "human":
            self._render_frame()

        return observation_clever_pop, info
    
    def step(self, clever_pop_actions):
        # By default, random actions
        actions = self.underlying_env.random_action_grid(self.underlying_env.batch_size, self.underlying_env.rows, self.underlying_env.cols, self.underlying_env.device)
        
        # Overwrite with the clever population's actions.
        actions[:, self.clever_pop_id] = clever_pop_actions
        
        # Perform the environment step using the full actions.
        # We assume your underlying step returns reward, done, info.
        rewards_per_pop, dones_per_batch, info = self.underlying_env.step(actions)
        
        # Get new observations for the intelligent agents.
        #observations_per_population = [self.underlying_env.get_deep_observations(i).cpu().numpy() for i in range(self.underlying_env.number_of_populations)]
        observation_clever_pop = self.underlying_env.get_deep_observations(self.clever_pop_id).cpu().numpy()
        
        # Similarly convert rewards and dones to NumPy.
        reward_numpy = rewards_per_pop.cpu().numpy()
        
        done_numpy = dones_per_batch.cpu().numpy()
        
        # Gymnasium step() now should return:
        # obs, reward, terminated, truncated, info.
        terminated = done_numpy
        truncated = False   # can be customized based on a time limit, etc.
        
        return observation_clever_pop, reward_numpy, terminated, truncated, info

    def render(self, mode="human", batch_id=0, pause_time_s=0.05):
        """
        Render the current state of the simulation.
        For now, this displays the grid for batch 0.
        
        :param mode: "human" to open/update a window,
                     "rgb_array" to return a NumPy array with the image.
        :return: If mode == "rgb_array", returns the image as a NumPy array.
                 Otherwise returns None.
        """
        # Define population_colors: index 0 for unoccupied, and then for each population.
        population_colors = ['white', 'red', 'blue', 'green']
        cmap = mcolors.ListedColormap(population_colors)
        
        # --- Compute labels for batch 0 ---
        # Here we follow the logic from get_batch0_labels.
        # Assuming self.underlying_env.grid is a torch tensor of shape (nb_batches, nb_pop, rows, cols)
        grid0 = self.underlying_env.grid[batch_id].cpu().numpy()  # shape: (nb_pop, rows, cols)
        
        # For each cell, choose the population with maximum value.
        # Shift labels by +1 so populations are 1,2,...; unoccupied cells will be fixed next.
        labels = grid0.argmax(axis=0) + 1
        
        # Mark cells that are unoccupied (all channels sum to 0) as label 0.
        empty_mask = grid0.sum(axis=0) == 0
        labels[empty_mask] = 0
        
        # --- Rendering based on mode ---
        if mode == "rgb_array":
            # Create a temporary figure to render the image.
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(population_colors)-0.5)
            ax.axis('off')
            fig.canvas.draw()
            
            # Convert the rendered figure to a numpy RGB array.
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

        elif mode == "human":
            # Reuse or create a persistent figure.
            if not hasattr(self, "_render_fig"):
                self._render_fig, self._render_ax = plt.subplots()
            self._render_ax.clear()
            self._render_ax.imshow(labels, cmap=cmap, vmin=-0.5, vmax=len(population_colors)-0.5)
            self._render_ax.axis('off')
            self._render_ax.set_title("Batch 0 Simulation Grid")
            
            # Use pause to update the figure without blocking.
            plt.pause(pause_time_s)
            plt.show(block=False)
            return None
        else:
            raise NotImplementedError(f"Render mode {mode} is not implemented.")

    def close(self):
        if hasattr(self, "_render_fig"):
            plt.close(self._render_fig)
