QT = core

CONFIG += c++17 cmdline

# Target configuration
TARGET = car-controls-qt
TEMPLATE = app

# ====== PROJECT STRUCTURE ======
ZEROMQ_PATH = $$PWD/../../ZeroMQ
INCLUDES_PATH = $$PWD/includes
SOURCES_PATH = $$PWD/sources

# Include Paths (explicit inheritance from root)
INCLUDEPATH += \
	$$ZEROMQ_PATH \
	$$INCLUDES_PATH \
	$$INCLUDES_PATH/inference \
	$$INCLUDES_PATH/objectDetection

# Application Sources
SOURCES += \
	$$SOURCES_PATH/AutonomousMode.cpp \
	$$SOURCES_PATH/ControlsManager.cpp \
	$$SOURCES_PATH/Debugger.cpp \
	$$SOURCES_PATH/EngineController.cpp \
	$$SOURCES_PATH/JoysticksController.cpp \
	$$SOURCES_PATH/MPCOptimizer.cpp \
	$$SOURCES_PATH/MPCPlanner.cpp \
	$$SOURCES_PATH/PeripheralController.cpp \
	$$SOURCES_PATH/inference/CameraStreamer.cpp \
	$$SOURCES_PATH/inference/LaneCurveFitter.cpp \
	$$SOURCES_PATH/inference/LanePostProcessor.cpp \
	$$SOURCES_PATH/inference/TensorRTInferencer.cpp \
	$$SOURCES_PATH/main.cpp \
	$$SOURCES_PATH/objectDetection/LabelManager.cpp \
	$$SOURCES_PATH/objectDetection/YOLOv5TRT.cpp \
	$$ZEROMQ_PATH/Publisher.cpp \
	$$ZEROMQ_PATH/Subscriber.cpp \

HEADERS += \
	$$INCLUDES_PATH/AutonomousMode.hpp \
	$$INCLUDES_PATH/CommonTypes.hpp \
	$$INCLUDES_PATH/ControlsManager.hpp \
	$$INCLUDES_PATH/Debugger.hpp \
	$$INCLUDES_PATH/EngineController.hpp \
	$$INCLUDES_PATH/IPeripheralController.hpp \
	$$INCLUDES_PATH/JoysticksController.hpp \
	$$INCLUDES_PATH/MPCConfig.hpp \
	$$INCLUDES_PATH/MPCOptimizer.hpp \
	$$INCLUDES_PATH/MPCPlanner.hpp \
	$$INCLUDES_PATH/PeripheralController.hpp \
	$$INCLUDES_PATH/enums.hpp \
	$$INCLUDES_PATH/inference/CameraStreamer.hpp \
	$$INCLUDES_PATH/inference/IInferencer.hpp \
	$$INCLUDES_PATH/inference/LaneCurveFitter.hpp \
	$$INCLUDES_PATH/inference/LanePostProcessor.hpp \
	$$INCLUDES_PATH/inference/TensorRTInferencer.hpp \
	$$INCLUDES_PATH/objectDetection/LabelManager.hpp \
	$$INCLUDES_PATH/objectDetection/YOLOv5TRT.hpp \
	$$ZEROMQ_PATH/Publisher.hpp \
	$$ZEROMQ_PATH/Subscriber.hpp \

# Common Libraries
LIBS += -lSDL2 -lrt -lzmq -lnlopt -lmlpack -lboost_system -lstdc++fs

# Conditionally add paths for cross-compilation
contains(QT_ARCH, arm)|contains(QT_ARCH, arm64)|contains(QT_ARCH, aarch64) {
	LIBS += -L$$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu -lSDL2
	INCLUDEPATH += $$[QT_SYSROOT]/usr/include/SDL2

	message("Building for ARM architecture")

	QMAKE_SYSROOT = /home/michel/new_qtjetson/sysroot

	INCLUDEPATH += \
	$${QMAKE_SYSROOT}/../qt5.15/include \
	$${QMAKE_SYSROOT}/../qt5.15/include/QtCore \
	$${QMAKE_SYSROOT}/usr/include

	# CUDA includes
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/cuda/include
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/cuda-10.2/include
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/cuda-11.4/include
	
	# TensorRT includes
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/include
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/x86_64-linux-gnu
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/aarch64-linux-gnu

	# OpenCV includes
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/include/opencv4
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/opencv4

	# GStreamer includes
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/gstreamer-1.0
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/glib-2.0
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include

	# Eigen libraries
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include/eigen3

	# Library paths
	LIBS += -L$${QMAKE_SYSROOT}/usr/local/lib
	LIBS += -L$${QMAKE_SYSROOT}/usr/local/cuda-10.2/lib64
	LIBS += -L$${QMAKE_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas

	# TensorRT, CUDA, OpenCV
	LIBS += -lcudart -lnvinfer
	LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
	LIBS += -lopencv_dnn -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudacodec
	LIBS += -lcublasLt -llapack -lblas
	LIBS += -lnvmedia -lnvdla_compiler
	
	# OpenMP from sysroot to avoid GLIBC version conflicts
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/atlas
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9/libgomp.a

	# LAPACK and BLAS libraries
	LIBS += -llapack -lcblas -lblas -ltbb
	LIBS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu -lgfortran

	# GStreamer libraries
	LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

	# OpenGL, GLEW, GLFW libraries (ORDER MATTERS!)
	LIBS += -lGLEW -lglfw -lGL

	# ZeroMQ includes
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/include
	INCLUDEPATH += $${QMAKE_SYSROOT}/usr/local/include

	# RPath for custom OpenCV runtime
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/local/lib
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/local/cuda-10.2/lib64
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/9
	
	# Force using sysroot libraries for glibc compatibility
	QMAKE_LFLAGS += -Wl,-rpath-link,$${QMAKE_SYSROOT}/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -L$${QMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
	QMAKE_LFLAGS += -L$${QMAKE_SYSROOT}/lib/aarch64-linux-gnu
	
	# Static link with compatible libstdc++ to avoid glibc version conflicts
	QMAKE_LFLAGS += -static-libstdc++ -static-libgcc
}

# Configurações de debug/release
CONFIG(debug, debug|release) {
    message("Debug build")
    DEFINES += DEBUG_BUILD
    QMAKE_CXXFLAGS += -g -DDEBUG -fopenmp
} else {
    message("Release build")
    DEFINES += RELEASE_BUILD
    QMAKE_CXXFLAGS += -O3 -DNDEBUG -fopenmp
}

# Output directories
CONFIG(debug, debug|release) {
    DESTDIR = build/debug
} else {
    DESTDIR = build/release
}

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.rcc
UI_DIR = $$DESTDIR/.ui

# Adicionando flags de compilação para warnings e erros
# QMAKE_CXXFLAGS += -Wall -Werror -Wextra -pedantic
