QT       += core
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes/main \
    $$PWD/includes/data \
    $$PWD/includes/canbus \
    $$PWD/includes/controls \
    $$PWD/includes/display \
    $$PWD/includes/system \
    $$PWD/includes/utils \
    $$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
    $$TESTS_PATH/unit/test_SPIController.cpp \
    $$TESTS_PATH/integration/test_MCP2515Controller.cpp \
    sources/canbus/SPIController.cpp \
    sources/canbus/MCP2515Configurator.cpp  # Add this line

HEADERS += \
    $$TESTS_PATH/mocks/MockSPI.hpp \
    includes/canbus/MCP2515Configurator.hpp \
    includes/canbus/SPIController.hpp

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
