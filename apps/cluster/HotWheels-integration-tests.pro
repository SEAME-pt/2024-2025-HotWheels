QT       += core testlib network
CONFIG   += c++17

# ====== Integration Tests Target ======
TEMPLATE = app
TARGET = HotWheels-integration-tests

# Include Paths
INCLUDEPATH += \
	$$PWD/includes/data \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
	$$PWD/includes/system \
	$$PWD/includes/canbus

# Integration Test Sources
INTEGRATION_TESTS_PATH = app_tests/integration

SOURCES += \
	$$INTEGRATION_TESTS_PATH/test_int_SystemManager.cpp \
	$$INTEGRATION_TESTS_PATH/test_int_DataManager.cpp \
	$$INTEGRATION_TESTS_PATH/test_int_MileageManager.cpp \
	$$INTEGRATION_TESTS_PATH/test_int_CanBusManager.cpp

# System Sources Required for Tests
SOURCES += \
	sources/system/BatteryController.cpp \
	sources/system/SystemInfoProvider.cpp \
	sources/system/SystemCommandExecutor.cpp \
	sources/system/SystemManager.cpp \
	sources/utils/I2CController.cpp \
	sources/utils/FileController.cpp \
	sources/mileage/MileageManager.cpp \
	sources/mileage/MileageFileHandler.cpp \
	sources/mileage/MileageCalculator.cpp \
	sources/canbus/CanBusManager.cpp \
	sources/canbus/MCP2515Controller.cpp \
	sources/canbus/MCP2515Configurator.cpp \
	sources/canbus/CANMessageProcessor.cpp \
	sources/canbus/SPIController.cpp \
	sources/data/DataManager.cpp \
	sources/data/SystemDataManager.cpp \
	sources/data/ClusterSettingsManager.cpp \
	sources/data/VehicleDataManager.cpp

# Sytem includes Required for Tests
HEADERS += \
	includes/system/SystemManager.hpp \
	includes/mileage/MileageManager.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/data/DataManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
	includes/data/VehicleDataManager.hpp

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
