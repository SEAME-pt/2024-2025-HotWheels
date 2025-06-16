QT       += core testlib network widgets
CONFIG   += c++17

# ====== Unit Tests Target ======
TEMPLATE = app
TARGET = HotWheels-unit-tests

# Include Paths
INCLUDEPATH += \
	$$PWD/includes/canbus \
	$$PWD/includes/data \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
	$$PWD/includes/system \
	$$PWD/includes/display \
	$$PWD/app_tests/mocks

# Unit Test Sources
UNIT_TESTS_PATH = app_tests/unit
MOCKS_PATH = app_tests/mocks

SOURCES += \
	$$UNIT_TESTS_PATH/canbus/test_SPIController.cpp \
	$$UNIT_TESTS_PATH/canbus/test_MCP2515Configurator.cpp \
	$$UNIT_TESTS_PATH/canbus/test_CANMessageProcessor.cpp \
	$$UNIT_TESTS_PATH/canbus/test_MCP2515Controller.cpp \
	$$UNIT_TESTS_PATH/canbus/test_CanBusManager.cpp \
	$$UNIT_TESTS_PATH/data/test_SystemDataManager.cpp \
	$$UNIT_TESTS_PATH/data/test_VehicleDataManager.cpp \
	$$UNIT_TESTS_PATH/data/test_ClusterSettingsManager.cpp \
	$$UNIT_TESTS_PATH/mileage/test_MileageFileHandler.cpp \
	$$UNIT_TESTS_PATH/mileage/test_MileageCalculator.cpp \
	$$UNIT_TESTS_PATH/mileage/test_MileageManager.cpp \
	$$UNIT_TESTS_PATH/system/test_BatteryController.cpp \
	$$UNIT_TESTS_PATH/system/test_SystemInfoProvider.cpp \
	$$UNIT_TESTS_PATH/system/test_SystemManager.cpp

# Unit Test Headers (Mocks)
HEADERS += \
	$$MOCKS_PATH/MockSPIController.hpp \
	$$MOCKS_PATH/MockMCP2515Controller.hpp \
	$$MOCKS_PATH/MockFileController.hpp \
	$$MOCKS_PATH/MockMileageFileHandler.hpp \
	$$MOCKS_PATH/MockMileageCalculator.hpp \
	$$MOCKS_PATH/MockSystemCommandExecutor.hpp \
	$$MOCKS_PATH/MockSystemInfoProvider.hpp \
	$$MOCKS_PATH/MockBatteryController.hpp

# System Sources Required for Tests
SOURCES += \
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
	sources/data/VehicleDataManager.cpp \
	sources/display/DisplayManager.cpp \
	sources/display/NotificationManager.cpp \
	sources/display/NotificationOverlay.cpp

# Sytem includes Required for Tests
HEADERS += \
	includes/system/SystemManager.hpp \
	includes/mileage/MileageManager.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/canbus/MCP2515Configurator.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
	includes/data/VehicleDataManager.hpp \
	includes/display/DisplayManager.hpp \
	includes/display/NotificationManager.hpp \
	includes/display/NotificationOverlay.hpp


# Define paths for Jetson cross-compilation
JETSON_SYSROOT = /home/seame/qtjetson/sysroot

GMOCK_LIBDIR = $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
LIBS += -L$${GMOCK_LIBDIR} \
        -lgmock_main -lgtest_main -lgmock -lgtest -lpthread

