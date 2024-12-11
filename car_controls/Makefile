# Cross-compilation Makefile with SDL2 setup

# Cross-compilation toolchain
CXX = aarch64-linux-gnu-g++
CC = aarch64-linux-gnu-gcc

# Compiler and linker flags
CXXFLAGS = -march=armv8-a -mtune=cortex-a53
LDFLAGS = 

# SDL2 download and build variables
SDL2_VERSION = 2.28.5
SDL2_IMAGE_VERSION = 2.6.3
SDL2_BASE_URL = https://github.com/libsdl-org/SDL/releases/download/release-$(SDL2_VERSION)
SDL2_IMAGE_BASE_URL = https://github.com/libsdl-org/SDL_image/releases/download/release-$(SDL2_IMAGE_VERSION)
SDL2_BUILD_DIR = /tmp/sdl2-cross-compile

# Targets
TARGET = car_controls
SOURCES = main.cpp Jetcar.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: sdl2-setup $(TARGET)

# SDL2 cross-compilation setup
sdl2-setup:
	@echo "Setting up SDL2 for cross-compilation..."
	@mkdir -p $(SDL2_BUILD_DIR)
	@cd $(SDL2_BUILD_DIR) && \
	wget $(SDL2_BASE_URL)/SDL2-$(SDL2_VERSION).tar.gz && \
	wget $(SDL2_IMAGE_BASE_URL)/SDL2_image-$(SDL2_IMAGE_VERSION).tar.gz && \
	tar -xzvf SDL2-$(SDL2_VERSION).tar.gz && \
	tar -xzvf SDL2_image-$(SDL2_IMAGE_VERSION).tar.gz

	@cd $(SDL2_BUILD_DIR)/SDL2-$(SDL2_VERSION) && \
	mkdir -p build && cd build && \
	cmake .. \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_INSTALL_PREFIX=/usr/aarch64-linux-gnu \
		-DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
		-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY && \
	make && \
	sudo make install

	@cd $(SDL2_BUILD_DIR)/SDL2_image-$(SDL2_IMAGE_VERSION) && \
	mkdir -p build && cd build && \
	cmake .. \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_INSTALL_PREFIX=/usr/aarch64-linux-gnu \
		-DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
		-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY && \
	make && \
	sudo make install

	@echo "SDL2 cross-compilation setup complete!"

# SDL2 flags for cross-compilation
SDL_CFLAGS = $(shell PKG_CONFIG_PATH=/usr/aarch64-linux-gnu/lib/pkgconfig pkg-config --cflags sdl2 SDL2_image)
SDL_LIBS = $(shell PKG_CONFIG_PATH=/usr/aarch64-linux-gnu/lib/pkgconfig pkg-config --libs sdl2 SDL2_image)

# Combine flags
CXXFLAGS += $(SDL_CFLAGS) -pthread
LDFLAGS += $(SDL_LIBS)

# Build the target
$(TARGET): $(OBJECTS)
	@echo "Linking for aarch64..."
	$(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(CXXFLAGS)
	@echo "Executable compiled!"

# Compile source files into object files
%.o: %.cpp
	@echo "Compiling $< for aarch64..."
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Clean targets
clean:
	rm -f $(OBJECTS)
	rm -rf $(SDL2_BUILD_DIR)

fclean: clean
	rm -f $(TARGET)

re: fclean all

# Diagnostic target
debug:
	@echo "Compiler: $(CXX)"
	@$(CXX) --version
	@echo "\nSDL2 Compilation Flags:"
	@PKG_CONFIG_PATH=/usr/aarch64-linux-gnu/lib/pkgconfig pkg-config --cflags --libs sdl2 SDL2_image
	@echo "\nLibrary Paths:"
	@$(CXX) -print-search-dirs
