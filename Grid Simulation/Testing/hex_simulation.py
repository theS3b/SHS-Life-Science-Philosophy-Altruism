import random

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

