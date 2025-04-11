QT	   += core gui network
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++17

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$PWD/includes/main \
	$$PWD/includes/data \
	$$PWD/includes/canbus \
	$$PWD/includes/controls \
	$$PWD/includes/display \
	$$PWD/includes/system \
	$$PWD/includes/mileage \
	$$PWD/includes/utils \
	$$PWD/includes/camera

# Application Sources
SOURCES += \
	../../ZeroMQ/Publisher.cpp \
	../../ZeroMQ/Subscriber.cpp \
	sources/main/main.cpp \
	sources/main/CarManager.cpp \
	sources/data/DataManager.cpp \
	sources/data/SystemDataManager.cpp \
	sources/data/VehicleDataManager.cpp \
	sources/data/ClusterSettingsManager.cpp \
	sources/canbus/MCP2515Controller.cpp \
	sources/canbus/CanBusManager.cpp \
	sources/canbus/SPIController.cpp \
	sources/canbus/MCP2515Configurator.cpp \
	sources/canbus/CANMessageProcessor.cpp \
	sources/controls/ControlsManager.cpp \
	sources/display/DisplayManager.cpp \
	sources/system/SystemManager.cpp \
	sources/system/SystemCommandExecutor.cpp \
	sources/system/SystemInfoProvider.cpp \
	sources/system/BatteryController.cpp \
	sources/mileage/MileageCalculator.cpp \
	sources/mileage/MileageManager.cpp \
	sources/mileage/MileageFileHandler.cpp \
	sources/utils/I2CController.cpp \
	sources/utils/FileController.cpp \
	sources/camera/CameraStreamer.cpp \
	sources/camera/TensorRTInferencer.cpp

HEADERS += \
	../../ZeroMQ/Publisher.hpp \
	../../ZeroMQ/Subscriber.hpp \
	includes/main/CarManager.hpp \
	includes/data/DataManager.hpp \
	includes/data/SystemDataManager.hpp \
	includes/data/VehicleDataManager.hpp \
	includes/data/ClusterSettingsManager.hpp \
	includes/data/enums.hpp \
	includes/canbus/MCP2515Controller.hpp \
	includes/canbus/IMCP2515Controller.hpp \
	includes/canbus/CanBusManager.hpp \
	includes/canbus/SPIController.hpp \
	includes/canbus/ISPIController.hpp \
	includes/canbus/MCP2515Configurator.hpp \
	includes/canbus/CANMessageProcessor.hpp \
	includes/controls/ControlsManager.hpp \
	includes/display/DisplayManager.hpp \
	includes/system/SystemManager.hpp \
	includes/system/BatteryController.hpp \
	includes/system/IBatteryController.hpp \
	includes/system/SystemCommandExecutor.hpp \
	includes/system/SystemInfoProvider.hpp \
	includes/system/ISystemCommandExecutor.hpp \
	includes/system/ISystemInfoProvider.hpp \
	includes/mileage/MileageCalculator.hpp \
	includes/mileage/MileageManager.hpp \
	includes/mileage/MileageFileHandler.hpp \
	includes/mileage/IMileageFileHandler.hpp \
	includes/utils/I2CController.hpp \
	includes/utils/II2CController.hpp \
	includes/utils/FileController.hpp \
	includes/camera/CameraStreamer.hpp \
	includes/camera/TensorRTInferencer.hpp

FORMS += forms/CarManager.ui

RESOURCES += \
	forms/resources.qrc

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm) {
	message("Building for ARM architecture")

	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	# Define paths for Jetson cross-compilation
	JETSON_SYSROOT = /home/michel/qtjetson/sysroot

	# CUDA includes - use the exact path found on Jetson
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include

	# TensorRT includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu

	# OpenCV includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/opencv4

	# GStreamer includes (needed for camera streaming)
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/gstreamer-1.0
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/glib-2.0
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include

	# Library paths for ARM - add all necessary paths
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra  # Add Tegra libraries path

	# Make sure we're linking against the right libraries
	LIBS += -lcudart -lnvinfer -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
	LIBS += -lnvmedia -lnvdla_compiler

	# GStreamer libraries
	LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

	# Add rpath to help find libraries at runtime
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra

}
