cmake_minimum_required(VERSION 3.14)
project(cppenv_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use conda Python (this works with conda-forge pybind11)
find_package(pybind11 REQUIRED)

# Include your simulation and bindings
include_directories(env/include)

add_library(cppenv MODULE
    env/src/hex_simulation.cpp
    bindings/pybind_wrapper.cpp
)

# Ensure correct module name
set_target_properties(cppenv PROPERTIES PREFIX "")
set_target_properties(cppenv PROPERTIES SUFFIX ".pyd")

# Link against pybind11
target_link_libraries(cppenv PRIVATE pybind11::module)
