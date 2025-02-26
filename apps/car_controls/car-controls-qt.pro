QT = core

CONFIG += c++17 cmdline

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
    $$PWD/includes \
    /usr/include/Ice \
    /usr/include/IceUtil \
    $$[QT_SYSROOT]/usr/include/Ice \
    $$[QT_SYSROOT]/usr/include/IceUtil


# Application Sources
SOURCES += \
    ../../ZeroC/CarDataI.cpp \
    ../../ZeroC/ClientThread.cpp \
    ../../ZeroC/Joystick.cpp \
    sources/ControlsManager.cpp \
    sources/JoysticksController.cpp \
    sources/EngineController.cpp \
    sources/PeripheralController.cpp \
    sources/main.cpp

HEADERS += \
    ../../ZeroC/CarDataI.hpp \
    ../../ZeroC/ClientThread.hpp \
    ../../ZeroC/Joystick.h \
    includes/ControlsManager.hpp \
    includes/JoysticksController.hpp \
    includes/EngineController.hpp \
    includes/PeripheralController.hpp \
    includes/IPeripheralController.hpp \
    includes/enums.hpp

# Common Libraries
LIBS += -lSDL2 -lrt -lIce

# Add explicit path for Ice library
LIBS += -L/home/michel/qt***/sysroot/usr/lib/aarch64-linux-gnu -lIce

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
    LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
    INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
}
