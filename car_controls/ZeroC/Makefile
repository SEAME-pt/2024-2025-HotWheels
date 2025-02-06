# Makefile for Ice-based server and client

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -I.

# Ice libraries and flags
LIBS = -lIce -lpthread

# Source files
SRCS_SERVER = server.cpp Joystick.cpp
SRCS_CLIENT = client.cpp Joystick.cpp

# Output files (executables)
OUT_SERVER = server
OUT_CLIENT = client

# Default target to build both server and client
all: $(OUT_SERVER) $(OUT_CLIENT)

# Rule to build the server
$(OUT_SERVER): $(SRCS_SERVER)
	$(CXX) $(CXXFLAGS) -o $(OUT_SERVER) $(SRCS_SERVER) $(LIBS)

# Rule to build the client
$(OUT_CLIENT): $(SRCS_CLIENT)
	$(CXX) $(CXXFLAGS) -o $(OUT_CLIENT) $(SRCS_CLIENT) $(LIBS)

# Clean rule to remove compiled files
clean:
	rm -f $(OUT_SERVER) $(OUT_CLIENT)

# Phony targets
.PHONY: all clean
