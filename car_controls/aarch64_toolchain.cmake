# Specify the toolchain file for cross-compilation to aarch64
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler paths
set(CMAKE_C_COMPILER /usr/aarch64-linux-gnu/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/aarch64-linux-gnu/bin/g++)

# Include and link directories for aarch64
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# Include paths for libraries and headers
include_directories(SYSTEM /usr/aarch64-linux-gnu/include)
link_directories(SYSTEM /usr/aarch64-linux-gnu/lib)
