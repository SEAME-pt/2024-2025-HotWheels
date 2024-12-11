cmake_minimum_required(VERSION 3.10)

# Project name and version
project(CarControls VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add executable
add_executable(car_controls main.cpp Jetcar.cpp)

# Find SDL2
find_package(SDL2 REQUIRED)

# Link SDL2
target_link_libraries(car_controls SDL2)
