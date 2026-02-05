cmake_minimum_required(VERSION 3.10)

project(Transformers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE DEBUG)

add_library(Transformers STATIC)

target_sources(Transformers
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/graph.cpp
  
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/tensor.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/graph.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/transformers.hpp
)

target_include_directories(Transformers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

set_target_properties(Transformers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS Transformers
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/include/
    DESTINATION include/transformers
    FILES_MATCHING PATTERN "*.hpp"
)

enable_testing()
add_subdirectory(tests)
