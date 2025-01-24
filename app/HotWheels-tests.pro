QT       += core
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes/canbus \
    $$PWD/includes/controls \
    $$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
    $$TESTS_PATH/unit/canbus/test_SPIController.cpp \
    sources/canbus/SPIController.cpp \
    $$TESTS_PATH/unit/controls/test_PeripheralController.cpp \
    sources/controls/PeripheralController.cpp \

HEADERS += \
    $$TESTS_PATH/mocks/MockSPI.hpp \
    $$TESTS_PATH/mocks/MockPeripheralController.hpp \

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
