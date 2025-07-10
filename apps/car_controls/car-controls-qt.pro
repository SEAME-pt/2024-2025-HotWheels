QT = core

CONFIG += c++17 cmdline

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$PWD/includes \
	$$PWD/includes/inference


# Application Sources
SOURCES += \
	../../ZeroMQ/Publisher.cpp \
	../../ZeroMQ/Subscriber.cpp \
	sources/inference/CameraStreamer.cpp \
	sources/inference/TensorRTInferencer.cpp \
	sources/inference/LanePostProcessor.cpp \
	sources/inference/LaneCurveFitter.cpp \
	sources/objectDetection/LabelManager.cpp \
	sources/objectDetection/YOLOv5TRT.cpp \
	sources/ControlsManager.cpp \
	sources/JoysticksController.cpp \
	sources/EngineController.cpp \
	sources/PeripheralController.cpp \
	sources/main.cpp

HEADERS += \
	../../ZeroMQ/Publisher.hpp \
	../../ZeroMQ/Subscriber.hpp \
	includes/inference/CameraStreamer.hpp \
	includes/inference/TensorRTInferencer.hpp \
	includes/inference/IInferencer.hpp \
	includes/inference/LanePostProcessor.hpp \
	includes/inference/LaneCurveFitter.hpp \
	includes/inference/Logger.hpp \
	includes/objectDetection/LabelManager.hpp \
	includes/objectDetection/YOLOv5TRT.hpp \
	includes/ControlsManager.hpp \
	includes/JoysticksController.hpp \
	includes/EngineController.hpp \
	includes/PeripheralController.hpp \
	includes/IPeripheralController.hpp \
	includes/enums.hpp

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	message("Building for ARM architecture")

	JETSON_SYSROOT = /home/seame/new_qtjetson/sysroot

	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include

	# CUDA includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include

	# TensorRT includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu

	# OpenCV includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/local/include/opencv4
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/opencv4

	# GStreamer includes
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/gstreamer-1.0
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/glib-2.0
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include

	# OpenGL, GLFW, GLEW includes
	INCLUDEPATH += /usr/local/include
	INCLUDEPATH += /usr/include/GL
	INCLUDEPATH += /usr/include/GLFW

	# Library paths
	LIBS += -L$${JETSON_SYSROOT}/usr/local/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/lib64
	LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

	# Eigen libraries
	INCLUDEPATH += $${JETSON_SYSROOT}/usr/include/eigen3

	# TensorRT, CUDA, OpenCV
	LIBS += -lcudart -lnvinfer
	LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
	LIBS += -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudacodec
	LIBS += -lcublasLt -lnvmedia -lnvdla_compiler

	# GStreamer libraries
	LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

	# OpenGL, GLEW, GLFW libraries (ORDER MATTERS!)
	LIBS += -lGLEW -lglfw -lGL

	# LAPACK and BLAS libraries (required by OpenCV)
	# Use specific gfortran version that matches target system
	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/atlas  # Add ATLAS BLAS/LAPACK path
	LIBS += -llapack -lcblas -lblas -ltbb

	LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu -lstdc++

	# RPath for custom OpenCV runtime
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/lib
	QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra

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
