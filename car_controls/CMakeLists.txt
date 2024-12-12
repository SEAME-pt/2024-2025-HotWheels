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

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Add the executable
add_executable(jetcar ${SOURCES})

# Link SDL2
target_link_libraries(jetcar ${SDL2_LIBRARIES})
