QT       += core
CONFIG   += c++17
TARGET   = HotWheels-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes/main \
    $$PWD/includes/data \
    $$PWD/includes/canbus \
    $$PWD/includes/controls \
    $$PWD/includes/display \
    $$PWD/includes/system \
    $$PWD/includes/utils \
    $$PWD/app_tests/mocks

# Test Sources
TESTS_PATH = app_tests

SOURCES += \
    $$TESTS_PATH/unit/test_SPIController.cpp \
    sources/canbus/SPIController.cpp \
    $$TESTS_PATH/unit/test_EngineController.cpp \
    sources/controls/EngineController.cpp \
    sources/controls/PeripheralController.cpp \
    $$OUT_PWD/moc_EngineController.cpp \
    $$OUT_PWD/moc_PeripheralController.cpp


HEADERS += \
    $$TESTS_PATH/mocks/MockSPI.hpp \
    $$TESTS_PATH/mocks/MockEngine.hpp \

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
