# CircularMeterModule.pro

TEMPLATE = lib
CONFIG += staticlib
TARGET = CircularMeterModule

QT += quick

CONFIG += c++17

SOURCES += \
    sources/MeterController.cpp

HEADERS += \
    includes/MeterController.hpp

RESOURCES += \
    qml/qml.qrc

INCLUDEPATH += includes

DESTDIR = $$OUT_PWD
