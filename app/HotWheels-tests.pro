QT       += core testlib
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
	$$PWD/includes/canbus \
	$$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
	$$TESTS_PATH/unit/canbus/test_SPIController.cpp \
	$$TESTS_PATH/unit/canbus/test_MCP2515Configurator.cpp \
	$$TESTS_PATH/unit/canbus/test_CANMessageProcessor.cpp \
	$$TESTS_PATH/unit/canbus/test_MCP2515Controller.cpp \
	$$TESTS_PATH/unit/canbus/test_CanBusManager.cpp \
	sources/canbus/MCP2515Configurator.cpp \
	sources/canbus/CANMessageProcessor.cpp \
	sources/canbus/MCP2515Controller.cpp \
	sources/canbus/SPIController.cpp \
	sources/canbus/CanBusManager.cpp

HEADERS += \
	$$TESTS_PATH/mocks/MockSPIController.hpp \
	$$TESTS_PATH/mocks/MockMCP2515Controller.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/CanBusManager.hpp \

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
