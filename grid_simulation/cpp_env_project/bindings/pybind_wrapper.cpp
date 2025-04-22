#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hex_simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cppenv, m) {
    // Expose the Cell structure.
    py::class_<Cell>(m, "Cell")
        .def_readonly("pop", &Cell::pop)
        .def_readonly("v", &Cell::v)
        .def_readonly("p", &Cell::p);

    // Expose the Population structure.
    py::class_<Population>(m, "Population")
        .def_readonly("p", &Population::p)
        .def_readonly("mean_v", &Population::mean_v)
        .def_readonly("std_v", &Population::std_v);

    // Expose the HexSimulation class.
    py::class_<HexSimulation>(m, "HexSimulation")
        .def(py::init<int, int, const std::unordered_map<std::string, Population>&>())
        .def("get_neighbors", &HexSimulation::get_neighbors)
        .def("colonization_phase", &HexSimulation::colonization_phase)
        .def("conflict_phase", &HexSimulation::conflict_phase)
        .def("step", &HexSimulation::step)
        .def("run", &HexSimulation::run)
        .def("print_grid", &HexSimulation::print_grid);
}
