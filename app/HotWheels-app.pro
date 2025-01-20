QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
	sources/main/CarManager.cpp \
	sources/data/DataManager.cpp \
	sources/canbus/MCP2515Controller.cpp \
	sources/canbus/CanBusManager.cpp \
	sources/controls/ControlsManager.cpp \
	sources/controls/JoysticksController.cpp \
	sources/controls/EngineController.cpp \
	sources/display/DisplayManager.cpp \
	sources/display/ButtonsController.cpp \
	sources/system/SystemManager.cpp \
	sources/system/BatteryController.cpp \
	sources/utils/I2CController.cpp \
	sources/main/main.cpp

HEADERS += \
	includes/data/DataManager.hpp \
	includes/data/enums.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/controls/ControlsManager.hpp \
	includes/controls/JoysticksController.hpp \
	includes/controls/EngineController.hpp \
	includes/display/DisplayManager.hpp \
	includes/display/ButtonsController.hpp \
	includes/system/SystemManager.hpp \
	includes/system/BatteryController.hpp \
	includes/utils/I2CController.hpp \
	includes/main/CarManager.hpp

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
