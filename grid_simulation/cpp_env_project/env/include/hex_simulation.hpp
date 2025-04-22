#ifndef HEX_SIMULATION_HPP
#define HEX_SIMULATION_HPP

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <random>

/* USE WITH:
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
*/

/// A cell on the hex grid.
struct Cell {
    std::string pop;  // population identifier
    double v;         // selective value
    double p;         // colonization/altruism probability
};

/// Population parameters.
struct Population {
    double p;
    double mean_v;
    double std_v;
};

/// Hexagonal grid simulation.
class HexSimulation {
public:
    /// Constructor.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param populations Mapping from population id to parameters.
    HexSimulation(int rows, int cols, const std::unordered_map<std::string, Population>& populations);

    /// Get neighbor coordinates for cell at (r,c).
    /// @param r Row.
    /// @param c Column.
    /// @param border_connected Whether grid borders wrap.
    std::vector<std::pair<int, int>> get_neighbors(int r, int c, bool border_connected = true);

    /// Colonization phase: empty cells may be colonized.
    void colonization_phase();

    /// Conflict phase: process conflicts between cells.
    void conflict_phase();

    /// One full simulation iteration.
    void step();

    /// Run for a given number of iterations.
    void run(int iterations);

    /// Print the grid to standard output.
    void print_grid() const;

private:
    int rows;
    int cols;
    std::vector<std::vector<std::optional<Cell>>> grid;
    std::unordered_map<std::string, Population> populations;
    std::mt19937 rng;
};

#endif // HEX_SIMULATION_HPP
