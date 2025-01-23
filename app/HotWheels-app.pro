QT       += core gui
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
    $$PWD/includes/utils

# Application Sources
SOURCES += \
    sources/main/main.cpp \
    sources/main/CarManager.cpp \
    sources/data/DataManager.cpp \
    sources/canbus/MCP2515Controller.cpp \
    sources/canbus/CanBusManager.cpp \
    sources/canbus/SPIController.cpp \
    sources/canbus/MCP2515Configurator.cpp \
    sources/canbus/CANMessageProcessor.cpp \
    sources/controls/ControlsManager.cpp \
    sources/controls/JoysticksController.cpp \
    sources/controls/EngineController.cpp \
    sources/controls/PeripheralController.cpp \
    sources/display/DisplayManager.cpp \
    sources/system/SystemManager.cpp \
    sources/system/BatteryController.cpp \
    sources/utils/I2CController.cpp

HEADERS += \
    includes/main/CarManager.hpp \
    includes/data/DataManager.hpp \
    includes/data/enums.hpp \
    includes/canbus/MCP2515Controller.hpp \
    includes/canbus/CanBusManager.hpp \
    includes/canbus/SPIController.hpp \
    includes/canbus/MCP2515Configurator.hpp \
    includes/canbus/CANMessageProcessor.hpp \
    includes/controls/ControlsManager.hpp \
    includes/controls/JoysticksController.hpp \
    includes/controls/EngineController.hpp \
    includes/controls/PeripheralController.hpp \
    includes/display/DisplayManager.hpp \
    includes/system/SystemManager.hpp \
    includes/system/BatteryController.hpp \
    includes/utils/I2CController.hpp

FORMS += forms/CarManager.ui

RESOURCES += \
	forms/resources.qrc

# Common Libraries
LIBS += -lSDL2

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
    LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
    INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
}
