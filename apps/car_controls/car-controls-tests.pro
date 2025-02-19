QT       += core testlib
CONFIG   += c++17
TARGET   = car-controls-tests

# Include Paths
INCLUDEPATH += \
    $$PWD/includes
    $$PWD/tests/mocks
# Test Sources
TESTS_PATH = tests

SOURCES += \
  $$TESTS_PATH/unit/test_PeripheralController.cpp \
  sources/PeripheralController.cpp

HEADERS += \
  $$TESTS_PATH/mocks/MockPeripheralController.hpp

# Link GTest and GMock
LIBS += -lgmock_main -lgtest_main -lpthread -lgmock -lgtest
