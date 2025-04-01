#include "hex_simulation.hpp"
#include <iostream>
#include <random>
#include <algorithm>

// Constructor: initialize grid and random engine.
HexSimulation::HexSimulation(int rows, int cols, const std::unordered_map<std::string, Population>& populations)
    : rows(rows), cols(cols), populations(populations) {
    grid.resize(rows, std::vector<std::optional<Cell>>(cols, std::nullopt));
    std::random_device rd;
    rng.seed(rd());
}

std::vector<std::pair<int, int>> HexSimulation::get_neighbors(int r, int c, bool border_connected) {
    std::vector<std::pair<int, int>> neighbors;
    std::vector<std::pair<int, int>> directions;
    if (r % 2 == 0) {
        directions = { {-1,-1}, {-1,0}, {0,-1}, {0,1}, {1,-1}, {1,0} };
    } else {
        directions = { {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,0}, {1,1} };
    }
    for (auto& d : directions) {
        int nr = r + d.first;
        int nc = c + d.second;
        if (border_connected) {
            nr = (nr + rows) % rows;
            nc = (nc + cols) % cols;
        }
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
            neighbors.push_back({nr, nc});
        }
    }
    return neighbors;
}

void HexSimulation::colonization_phase() {
    // Store new colonizations as tuples (row, col, new Cell)
    std::vector<std::tuple<int, int, Cell>> new_cells;
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!grid[r][c].has_value()) {
                auto neigh_coords = get_neighbors(r, c);
                // Map population id to neighboring cells of that type.
                std::unordered_map<std::string, std::vector<Cell>> pop_neighbors;
                for (auto& coord : neigh_coords) {
                    int nr = coord.first;
                    int nc = coord.second;
                    if (grid[nr][nc].has_value()) {
                        std::string pop = grid[nr][nc]->pop;
                        pop_neighbors[pop].push_back(grid[nr][nc].value());
                    }
                }
                std::vector<std::string> colonizer_candidates;
                for (auto& entry : pop_neighbors) {
                    const std::string& pop = entry.first;
                    const auto& cells = entry.second;
                    double sum_v = 0.0;
                    for (auto& cell : cells) {
                        sum_v += cell.v;
                    }
                    double prob = 0.0;
                    if (sum_v > 0.0) {
                        prob = 1 - 1 / sum_v;
                        if (prob < 0.0) prob = 0.0;
                        if (prob > 1.0) prob = 1.0;
                    }
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    if (dist(rng) < prob) {
                        colonizer_candidates.push_back(pop);
                    }
                }
                if (!colonizer_candidates.empty()) {
                    std::uniform_int_distribution<size_t> idx_dist(0, colonizer_candidates.size() - 1);
                    std::string chosen_pop = colonizer_candidates[idx_dist(rng)];
                    double mean_v = populations[chosen_pop].mean_v;
                    double std_v = populations[chosen_pop].std_v;
                    std::normal_distribution<double> normal_dist(mean_v, std_v);
                    double sampled_v = normal_dist(rng);
                    if (sampled_v <= 0)
                        sampled_v = 0.01;
                    double p_val = populations[chosen_pop].p;
                    Cell new_cell { chosen_pop, sampled_v, p_val };
                    new_cells.push_back(std::make_tuple(r, c, new_cell));
                }
            }
        }
    }
    
    // Apply new colonizations.
    for (auto& tup : new_cells) {
        int r, c;
        Cell cell;
        std::tie(r, c, cell) = tup;
        grid[r][c] = cell;
    }
}

void HexSimulation::conflict_phase() {
    double bonus_total = 0.5;
    // For each cell as potential defender.
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!grid[r][c].has_value())
                continue;
            Cell defender = grid[r][c].value();
            // Group enemy neighbors (cells with different population).
            std::unordered_map<std::string, std::vector<std::tuple<int, int, Cell>>> enemy_groups;
            auto neighbors = get_neighbors(r, c);
            for (auto& coord : neighbors) {
                int nr = coord.first;
                int nc = coord.second;
                if (grid[nr][nc].has_value() && grid[nr][nc]->pop != defender.pop) {
                    enemy_groups[grid[nr][nc]->pop].push_back(std::make_tuple(nr, nc, grid[nr][nc].value()));
                }
            }
            // Process conflict for enemy groups with at least two neighbors.
            for (auto& entry : enemy_groups) {
                const std::string& enemy_pop = entry.first;
                auto& attackers = entry.second;
                if (attackers.size() < 2)
                    continue;
                double p_attack = populations[enemy_pop].p;
                std::shuffle(attackers.begin(), attackers.end(), rng);
                for (auto& tup : attackers) {
                    int nr, nc;
                    Cell attacker;
                    std::tie(nr, nc, attacker) = tup;
                    // Proceed only if both defender and attacker still exist.
                    if (!grid[r][c].has_value() || !grid[nr][nc].has_value())
                        break;
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    if (dist(rng) < p_attack) {
                        double attacker_v = grid[nr][nc]->v;
                        double defender_v = grid[r][c]->v;
                        double new_attacker_v = attacker_v - defender_v;
                        double new_defender_v = defender_v - attacker_v;
                        grid[nr][nc]->v = new_attacker_v;
                        grid[r][c]->v = new_defender_v;
                        if (new_attacker_v <= 0)
                            grid[nr][nc] = std::nullopt;
                        if (new_defender_v <= 0) {
                            grid[r][c] = std::nullopt;
                            // Distribute bonus among enemy neighbors.
                            auto bonus_neighbors = get_neighbors(r, c);
                            std::vector<std::pair<int, int>> bonus_coords;
                            for (auto& coord2 : bonus_neighbors) {
                                int ar = coord2.first;
                                int ac = coord2.second;
                                if (grid[ar][ac].has_value() && grid[ar][ac]->pop == enemy_pop) {
                                    bonus_coords.push_back({ar, ac});
                                }
                            }
                            if (!bonus_coords.empty()) {
                                double bonus_each = bonus_total / bonus_coords.size();
                                for (auto& coord_bonus : bonus_coords) {
                                    int ar = coord_bonus.first;
                                    int ac = coord_bonus.second;
                                    if (grid[ar][ac].has_value())
                                        grid[ar][ac]->v += bonus_each;
                                }
                            }
                            break; // Stop processing further attackers for this defender.
                        }
                    }
                }
            }
        }
    }
}

void HexSimulation::step() {
    colonization_phase();
    conflict_phase();
}

void HexSimulation::run(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        step();
    }
}

void HexSimulation::print_grid() const {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!grid[r][c].has_value()) {
                std::cout << ".\t";
            } else {
                const Cell& cell = grid[r][c].value();
                // Print first character of the population and selective value.
                std::cout << cell.pop[0] << "(" << cell.v << ")\t";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
