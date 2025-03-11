QT       += core gui network
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++17

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$PWD/includes/main \
	$$PWD/includes/data \
	$$PWD/includes/canbus \
	$$PWD/includes/controls \
	$$PWD/includes/display \
	$$PWD/includes/system \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
	/usr/include/Ice \
        /usr/include/IceUtil \
	$$[QT_SYSROOT]/usr/include/Ice \
        $$[QT_SYSROOT]/usr/include/IceUtil

# Application Sources
SOURCES += \
        ../../ZeroC/CarDataI.cpp \
        ../../ZeroC/ClientThread.cpp \
        ../../ZeroC/Joystick.cpp \
        sources/main/main.cpp \
        sources/main/CarManager.cpp \
        sources/data/DataManager.cpp \
	sources/data/SystemDataManager.cpp \
	sources/data/VehicleDataManager.cpp \
	sources/data/ClusterSettingsManager.cpp \
        sources/canbus/MCP2515Controller.cpp \
        sources/canbus/CanBusManager.cpp \
        sources/canbus/SPIController.cpp \
        sources/canbus/MCP2515Configurator.cpp \
        sources/canbus/CANMessageProcessor.cpp \
        sources/controls/ControlsManager.cpp \
        sources/display/DisplayManager.cpp \
        sources/system/SystemManager.cpp \
	sources/system/SystemCommandExecutor.cpp \
	sources/system/SystemInfoProvider.cpp \
        sources/system/BatteryController.cpp \
	sources/mileage/MileageCalculator.cpp \
	sources/mileage/MileageManager.cpp \
	sources/mileage/MileageFileHandler.cpp \
	sources/utils/I2CController.cpp \
	sources/utils/FileController.cpp

HEADERS += \
        ../../ZeroC/CarDataI.hpp \
        ../../ZeroC/ClientThread.hpp \
        ../../ZeroC/Joystick.h \
        includes/main/CarManager.hpp \
        includes/data/DataManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/VehicleDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
        includes/data/enums.hpp \
        includes/canbus/MCP2515Controller.hpp \
	includes/canbus/IMCP2515Controller.hpp \
        includes/canbus/CanBusManager.hpp \
        includes/canbus/SPIController.hpp \
	includes/canbus/ISPIController.hpp \
        includes/canbus/MCP2515Configurator.hpp \
        includes/canbus/CANMessageProcessor.hpp \
        includes/controls/ControlsManager.hpp \
        includes/display/DisplayManager.hpp \
        includes/system/SystemManager.hpp \
        includes/system/BatteryController.hpp \
	includes/system/IBatteryController.hpp \
	includes/system/SystemCommandExecutor.hpp \
	includes/system/SystemInfoProvider.hpp \
	includes/system/ISystemCommandExecutor.hpp \
	includes/system/ISystemInfoProvider.hpp \
	includes/mileage/MileageCalculator.hpp \
	includes/mileage/MileageManager.hpp \
	includes/mileage/MileageFileHandler.hpp \
	includes/mileage/IMileageFileHandler.hpp \
	includes/utils/I2CController.hpp \
	includes/utils/II2CController.hpp \
	includes/utils/FileController.hpp

FORMS += forms/CarManager.ui

RESOURCES += \
	forms/resources.qrc

# Common Libraries
LIBS += -lSDL2 -lrt -lIce

# Add explicit path for Ice library
LIBS += -L/home/michel/qt***/sysroot/usr/lib/aarch64-linux-gnu -lIce

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
}
