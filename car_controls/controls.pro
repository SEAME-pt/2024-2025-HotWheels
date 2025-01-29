QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++17

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
    $$PWD/includes

# Application Sources
SOURCES += \
    sources/ControlsManager.cpp \
    sources/JoysticksController.cpp \
    sources/EngineController.cpp \
    sources/PeripheralController.cpp \

HEADERS += \
    includes/ControlsManager.hpp \
    includes/JoysticksController.hpp \
    includes/EngineController.hpp \
    includes/PeripheralController.hpp \
    includes/IPeripheralController.hpp \

# Common Libraries
LIBS += -lSDL2

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
    LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
    INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
}
