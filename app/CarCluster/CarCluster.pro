# CarCluster.pro

TEMPLATE = app
CONFIG += c++17
TARGET = CarCluster

QT += core gui widgets quick quickwidgets serialbus serialport

SOURCES += \
    sources/main.cpp \
    sources/CarCluster.cpp \
    sources/DisplayManager.cpp \
	sources/CircularMeterSetup.cpp \
	sources/MCP2515.cpp \
	sources/CanReceiverWorker.cpp \
	sources/SystemInfoUtility.cpp

RESOURCES += ../CircularMeterModule/qml/qml.qrc

RESOURCES += \
	data/data.qrc


HEADERS += \
    includes/CarCluster.h \
    includes/DisplayManager.hpp \
	includes/CircularMeterSetup.hpp \
	includes/CanReceiverWorker.hpp \
	includes/MCP2515.hpp \
	includes/SystemInfoUtility.hpp

FORMS += \
    forms/CarCluster.ui

INCLUDEPATH += \
    ../CircularMeterModule/includes \
    includes

LIBS += $$OUT_PWD/../CircularMeterModule/libCircularMeterModule.a


