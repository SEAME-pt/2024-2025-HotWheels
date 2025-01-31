QT       += core testlib
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes/canbus \
    $$PWD/includes/controls \
	$$PWD/includes/data \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
    $$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
	# $$TESTS_PATH/unit/canbus/test_SPIController.cpp \
	# $$TESTS_PATH/unit/canbus/test_MCP2515Configurator.cpp \
	# $$TESTS_PATH/unit/canbus/test_CANMessageProcessor.cpp \
	# $$TESTS_PATH/unit/canbus/test_MCP2515Controller.cpp \
	# $$TESTS_PATH/unit/canbus/test_CanBusManager.cpp \
 #  $$TESTS_PATH/unit/controls/test_PeripheralController.cpp \
 #  $$TESTS_PATH/unit/data/test_SystemDataManager.cpp \
 #  $$TESTS_PATH/unit/data/test_VehicleDataManager.cpp \
 #  $$TESTS_PATH/unit/data/test_ClusterSettingsManager.cpp \
 #  $$TESTS_PATH/unit/mileage/test_MileageFileHandler.cpp \
 #  $$TESTS_PATH/unit/mileage/test_MileageCalculator.cpp \
  $$TESTS_PATH/unit/mileage/test_MileageManager.cpp \
  sources/utils/FileController.cpp \
  sources/mileage/MileageFileHandler.cpp \
  sources/mileage/MileageCalculator.cpp \
  sources/mileage/MileageManager.cpp \
  sources/controls/PeripheralController.cpp \
	sources/canbus/MCP2515Configurator.cpp \
	sources/canbus/CANMessageProcessor.cpp \
	sources/canbus/MCP2515Controller.cpp \
	sources/canbus/SPIController.cpp \
	sources/canbus/CanBusManager.cpp \
	sources/data/SystemDataManager.cpp \
	sources/data/ClusterSettingsManager.cpp \
	sources/data/VehicleDataManager.cpp

HEADERS += \
	$$TESTS_PATH/mocks/MockSPIController.hpp \
	$$TESTS_PATH/mocks/MockMCP2515Controller.hpp \
  $$TESTS_PATH/mocks/MockPeripheralController.hpp \
  $$TESTS_PATH/mocks/MockFileController.hpp \
  $$TESTS_PATH/mocks/MockMileageFileHandler.hpp \
  $$TESTS_PATH/mocks/MockMileageCalculator.hpp \
  includes/mileage/MileageManager.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
	includes/data/VehicleDataManager.hpp

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
