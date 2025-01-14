# Variables
CXX = g++
CXXFLAGS = -lSDL2 -pthread
TARGET = aarch64-car-controls
SOURCES = main.cpp Jetcar.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)
# Build the target
$(TARGET): $(OBJECTS)
	@$(CXX) -o $(TARGET) $(OBJECTS) $(CXXFLAGS)
	@echo "Executable compiled!"

# Compile source files into object files
%.o: %.cpp
	@$(CXX) -c $< -o $@

# Clean target
clean:
	@rm -f $(OBJECTS)

fclean: clean
	@echo "Cleaning the executable..."
	@rm -f $(TARGET)
	@echo "Done!"

re: fclean all
