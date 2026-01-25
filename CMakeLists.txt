cmake_minimum_required(VERSION 3.10)

project(Transformers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE DEBUG)

add_library(Transformers STATIC)

target_sources(Transformers
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/layer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compute_graph.cpp
  
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/layer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/model.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/compute_graph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/transformers.hpp
)

target_include_directories(Transformers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

set_target_properties(Transformers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

enable_testing()
add_subdirectory(tests)
