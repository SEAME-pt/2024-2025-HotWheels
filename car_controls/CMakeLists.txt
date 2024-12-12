# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project information
project(JetcarProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    main.cpp
    Jetcar.cpp
)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Manually specify SDL2 include and library paths
set(SDL2_INCLUDE_DIR /usr/aarch64-linux-gnu/include/SDL2)
set(SDL2_LIBRARY /usr/aarch64-linux-gnu/lib/libSDL2.so)

include_directories(${SDL2_INCLUDE_DIR})
target_link_libraries(jetcar ${SDL2_LIBRARY})

# Add the executable
add_executable(jetcar ${SOURCES})

# Link SDL2
target_link_libraries(jetcar ${SDL2_LIBRARIES})
