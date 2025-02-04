QT       += core testlib
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes/canbus \
	$$PWD/includes/data \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
	$$PWD/includes/system \
	$$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
	# $$TESTS_PATH/integration/test_MCP2515Controller.cpp \
	$$TESTS_PATH/unit/canbus/test_SPIController.cpp \
	$$TESTS_PATH/unit/canbus/test_MCP2515Configurator.cpp \
	$$TESTS_PATH/unit/canbus/test_CANMessageProcessor.cpp \
	$$TESTS_PATH/unit/canbus/test_MCP2515Controller.cpp \
	$$TESTS_PATH/unit/canbus/test_CanBusManager.cpp \
  $$TESTS_PATH/unit/data/test_SystemDataManager.cpp \
  $$TESTS_PATH/unit/data/test_VehicleDataManager.cpp \
  $$TESTS_PATH/unit/data/test_ClusterSettingsManager.cpp \
  $$TESTS_PATH/unit/mileage/test_MileageFileHandler.cpp \
  $$TESTS_PATH/unit/mileage/test_MileageCalculator.cpp \
  $$TESTS_PATH/unit/mileage/test_MileageManager.cpp \
  $$TESTS_PATH/unit/system/test_BatteryController.cpp \
  $$TESTS_PATH/unit/system/test_SystemInfoProvider.cpp \
  $$TESTS_PATH/unit/system/test_SystemManager.cpp \
  sources/system/BatteryController.cpp \
  sources/system/SystemInfoProvider.cpp \
  sources/system/SystemCommandExecutor.cpp \
  sources/system/SystemManager.cpp \
  sources/utils/FileController.cpp \
  sources/utils/I2CController.cpp \
  sources/mileage/MileageFileHandler.cpp \
  sources/mileage/MileageCalculator.cpp \
  sources/mileage/MileageManager.cpp \
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
  $$TESTS_PATH/mocks/MockSystemCommandExecutor.hpp \
  $$TESTS_PATH/mocks/MockSystemInfoProvider.hpp \
  $$TESTS_PATH/mocks/MockBatteryController.hpp \
  includes/system/SystemManager.hpp \
  includes/mileage/MileageManager.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/canbus/MCP2515Configurator.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
	includes/data/VehicleDataManager.hpp

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
