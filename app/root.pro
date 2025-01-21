QT       += core gui widgets
CONFIG   += c++17

# Common Include Paths
INCLUDEPATH += \
    includes/main \
    includes/data \
    includes/canbus \
    includes/controls \
    includes/display \
    includes/system \
    includes/utils

# Common Libraries
LIBS += -lSDL2

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
    LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
    INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2
}

# ----------- HotWheels-app Configuration ----------- #
APP_TARGET = HotWheels-app
APP_SOURCES = \
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

APP_HEADERS = \
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

APP_FORMS = forms/CarManager.ui

# ----------- HotWheels-tests Configuration ----------- #
TEST_TARGET = HotWheels-tests
TESTS_PATH = app_tests

TEST_SOURCES = \
    $$TESTS_PATH/unit/test_SPIController.cpp \
    sources/canbus/SPIController.cpp

TEST_HEADERS = \
    $$TESTS_PATH/mocks/MockSPI.hpp \
    includes/canbus/SPIController.hpp

# Link GTest and GMock
TEST_LIBS = -lgmock_main -lgtest_main -lpthread -lgmock -lgtest

# ----------- Build Rules ----------- #
TEMPLATE = subdirs

SUBDIRS += app_target tests_target

app_target.file = $$PWD/HotWheels-app.pro
app_target.CONFIG += qt app
app_target.target = $$APP_TARGET
app_target.SOURCES += $$APP_SOURCES
app_target.HEADERS += $$APP_HEADERS
app_target.FORMS += $$APP_FORMS
app_target.LIBS += $$LIBS

tests_target.file = $$PWD/HotWheels-tests.pro
tests_target.CONFIG += qt app
tests_target.target = $$TEST_TARGET
tests_target.SOURCES += $$TEST_SOURCES
tests_target.HEADERS += $$TEST_HEADERS
tests_target.LIBS += $$TEST_LIBS
tests_target.INCLUDEPATH += $$INCLUDEPATH $$TESTS_PATH/mocks