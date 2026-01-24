cmake_minimum_required(VERSION 3.10)

project(Transformers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE DEBUG)
add_library(Transformers STATIC
    src/tensor.cpp
)

target_include_directories(Transformers PUBLIC src/include)

set_target_properties(Transformers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

enable_testing()
add_subdirectory(tests)
