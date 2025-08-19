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
	$$PWD/includes/camera \
	$$PWD/../../ZeroMQ

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
	sources/display/NotificationOverlay.cpp \
	sources/display/NotificationManager.cpp \
	sources/system/SystemManager.cpp \
	sources/system/SystemCommandExecutor.cpp \
	sources/system/SystemInfoProvider.cpp \
	sources/system/BatteryController.cpp \
	sources/mileage/MileageCalculator.cpp \
	sources/mileage/MileageManager.cpp \
	sources/mileage/MileageFileHandler.cpp \
	sources/utils/I2CController.cpp \
	sources/utils/FileController.cpp \

HEADERS += \
	../../ZeroMQ/Publisher.hpp \
	../../ZeroMQ/Subscriber.hpp \
	../../ZeroMQ/CommonTypes.hpp \
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
	includes/display/NotificationOverlay.hpp \
	includes/display/NotificationManager.hpp \
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

FORMS += forms/CarManager.ui

RESOURCES += \
	forms/resources.qrc

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq -lpthread

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	# Define paths for Jetson cross-compilation
	JETSON_SYSROOT = /home/seame/new_qtjetson/sysroot

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
	LIBS += -L$${JETSON_SYSROOT}/lib/aarch64-linux-gnu  # Add path for system libraries like pthread
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra  # Add Tegra libraries path
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/atlas  # Add ATLAS BLAS/LAPACK path
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9  # Add GCC runtime libs path

	# Make sure we're linking against the right libraries
	LIBS += -lcudart -lnvinfer -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
	LIBS += -lnvmedia -lnvdla_compiler

	# LAPACK and BLAS libraries (required by OpenCV)
	# Use specific gfortran version that matches target system
	LIBS += -llapack -lcblas -lblas -ltbb
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu -lgfortran

	# GStreamer libraries
	LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

	# Add rpath to help find libraries at runtime
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/atlas
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9
	QMAKE_LFLAGS += -Wl,-rpath,/usr/local/qt5.15/lib

	# Add runtime paths for target system
	QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/gcc/aarch64-linux-gnu/9

	# Add additional linker flags for compatibility
	QMAKE_LFLAGS += -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-in-shared-libs

	# Add static libstdc++ to avoid GLIBCXX version issues
	QMAKE_LFLAGS += -static-libstdc++
}
