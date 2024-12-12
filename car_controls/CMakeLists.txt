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

# Include project directories
include_directories(${CMAKE_SOURCE_DIR})

# Cross-compilation setup
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # Manually specify SDL2 include and library paths for AArch64
    set(SDL2_INCLUDE_DIR /usr/aarch64-linux-gnu/include/SDL2)
    set(SDL2_LIBRARY /usr/aarch64-linux-gnu/lib/libSDL2.so)

    include_directories(${SDL2_INCLUDE_DIR})
else()
    # Use find_package for native builds (optional)
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
endif()

# Add the executable
add_executable(car_controls ${SOURCES})

# Link SDL2
target_link_libraries(car_controls ${SDL2_LIBRARY})

