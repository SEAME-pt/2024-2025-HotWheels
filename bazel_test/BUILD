# BUILD file for your project

cc_binary(
    name = "aarch64-car-controls",  # Executable name
    srcs = [
        "main.cpp",  # Your source files
        "Jetcar.cpp",
        "Jetcar.hpp",
    ],
    deps = [],
    copts = [
        "-I/usr/include/SDL2",  # SDL2 headers location
        "-pthread",             # pthread flag
    ],
    linkopts = [
        "-L/usr/lib/aarch64-linux-gnu",  # SDL2 library location for aarch64
        "-lSDL2",  # Link against SDL2
    ],
)



