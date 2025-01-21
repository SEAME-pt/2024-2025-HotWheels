QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

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
    includes/display/DisplayManager.hpp \
    includes/system/SystemManager.hpp \
    includes/system/BatteryController.hpp \
    includes/utils/I2CController.hpp

FORMS += \
	forms/CarManager.ui

INCLUDEPATH += \
	includes/main \
	includes/data \
	includes/canbus \
	includes/controls \
	includes/display \
	includes/system \
	includes/utils

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
} else {
	LIBS += -lSDL2
}


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target