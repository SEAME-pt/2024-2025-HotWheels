# CarCluster.pro

TEMPLATE = app
CONFIG += c++17
TARGET = CarCluster

QT += core gui widgets quick quickwidgets serialbus serialport

SOURCES += \
    sources/main.cpp \
    sources/CarCluster.cpp \
    sources/DisplayManager.cpp \
	sources/CanController.cpp \
	sources/FakeSimulation.cpp \
	sources/SpiController.cpp \
	sources/CircularMeterSetup.cpp \
	sources/SystemInfoUtility.cpp

RESOURCES += ../CircularMeterModule/qml/qml.qrc

RESOURCES += \
	data/data.qrc


HEADERS += \
    includes/CarCluster.h \
    includes/DisplayManager.hpp \
	includes/CanController.hpp \
	includes/SpiController.hpp \
	includes/FakeSimulation.hpp \
	includes/CircularMeterSetup.hpp \
	includes/SystemInfoUtility.hpp

FORMS += \
    forms/CarCluster.ui

INCLUDEPATH += \
    ../CircularMeterModule/includes \
    includes

LIBS += $$OUT_PWD/../CircularMeterModule/libCircularMeterModule.a


