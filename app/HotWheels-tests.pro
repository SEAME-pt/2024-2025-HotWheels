QT       += core

CONFIG += c++17
TARGET = HotWheels-tests

TESTS_PATH = app_tests

SOURCES += \
    $$TESTS_PATH/unit/test_SPIController.cpp \
    sources/canbus/SPIController.cpp

HEADERS += \
    $$TESTS_PATH/mocks/MockSPI.hpp \
    includes/canbus/SPIController.hpp

INCLUDEPATH += \
    includes/main \
    includes/data \
    includes/canbus \
    includes/controls \
    includes/display \
    includes/system \
    includes/utils \
    $$TESTS_PATH/mocks

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest