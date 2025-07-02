cmake_minimum_required(VERSION 3.14)
project(TensorBindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate pybind11
find_package(pybind11 REQUIRED)

# Your library
add_library(redCedar SHARED tensor.cpp bindings.cpp nn.cpp)
target_include_directories(redCedar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(redCedar PRIVATE pybind11::module)
set_target_properties(redCedar PROPERTIES PREFIX "" SUFFIX ".so")
