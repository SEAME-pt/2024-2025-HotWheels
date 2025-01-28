# Variables
CXX = g++
CXXFLAGS = -lSDL2 -pthread
TARGET = aarch64-car-controls
SOURCES = main.cpp Jetcar.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Test variables
TEST_TARGET = tests/test_Jetcar
TEST_SOURCES = tests/test_Jetcar.cpp Jetcar.cpp
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# Google Test and Google Mock variables
GTEST_INCLUDE = /usr/include/gtest
GMOCK_INCLUDE = /usr/include/gmock
GTEST_LIB = /usr/lib/x86_64-linux-gnu
GMOCK_LIB = /usr/lib/x86_64-linux-gnu

GTEST_LIBS = -lgtest -lgmock -lgtest_main -pthread

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(OBJECTS)
	@$(CXX) -o $(TARGET) $(OBJECTS) $(CXXFLAGS)
	@echo "Executable compiled!"

# Build the test target
$(TEST_TARGET): $(TEST_OBJECTS)
	@$(CXX) -o $(TEST_TARGET) $(TEST_OBJECTS) $(CXXFLAGS) -I$(GTEST_INCLUDE) -I$(GMOCK_INCLUDE) -L$(GTEST_LIB) -L$(GMOCK_LIB) $(GTEST_LIBS)
	@echo "Test executable compiled!"

# Compile source files into object files
%.o: %.cpp
	@$(CXX) -c $< -o $@ -I$(GTEST_INCLUDE) -I$(GMOCK_INCLUDE)

# Clean target
clean:
	@rm -f $(OBJECTS) $(TEST_OBJECTS)

fclean: clean
	@echo "Cleaning the executables..."
	@rm -f $(TARGET) $(TEST_TARGET)
	@echo "Done!"

re: fclean all

# Run tests
test: $(TEST_TARGET)
	@./$(TEST_TARGET)

