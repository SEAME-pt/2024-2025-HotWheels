QT       += core testlib
CONFIG   += c++17
TARGET   = car-controls-tests

CONFIG += debug
QMAKE_CXXFLAGS += -fprofile-arcs -ftest-coverage -O0
QMAKE_LFLAGS   += -fprofile-arcs -ftest-coverage

JETSON_SYSROOT = /home/seame/new_qtjetson/sysroot

# Path to custom Linaro GCC 7.5 toolchain
LINARO_GCC7 = /home/seame/new_qtjetson/tools/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu

QMAKE_CC  = $$LINARO_GCC7/bin/aarch64-linux-gnu-gcc
QMAKE_CXX = $$LINARO_GCC7/bin/aarch64-linux-gnu-g++
QMAKE_LINK = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX

# Use the Jetson sysroot headers and libraries
QMAKE_CFLAGS   += --sysroot=$${JETSON_SYSROOT}
QMAKE_CXXFLAGS += --sysroot=$${JETSON_SYSROOT}
QMAKE_LFLAGS   += --sysroot=$${JETSON_SYSROOT}

QMAKE_CXXFLAGS += -fdebug-prefix-map=/home/seame/repos/cluster/apps=/home/jetson/apps/test

# Include Paths
INCLUDEPATH += \
    $$PWD/includes \
    $$PWD/tests/mocks \
    $$PWD/sources \
    $${JETSON_SYSROOT}/usr/local/include/opencv4 \
    $${JETSON_SYSROOT}/usr/include/opencv4 \
    $${JETSON_SYSROOT}/usr/local/cuda/include \
    $${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/include \
    $${JETSON_SYSROOT}/usr/include/aarch64-linux-gnu \
    $${JETSON_SYSROOT}/usr/include/gstreamer-1.0 \
    $${JETSON_SYSROOT}/usr/include/glib-2.0 \
    $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include \
    $${JETSON_SYSROOT}/usr/include/eigen3

# Test Sources
TESTS_PATH = tests

SOURCES += \
    $$TESTS_PATH/unit/test_PeripheralController.cpp \
    $$TESTS_PATH/unit/test_TensorRTInferencer.cpp \
    $$TESTS_PATH/unit/test_CameraStreamer.cpp \
    $$TESTS_PATH/unit/test_LabelManager.cpp \
    $$TESTS_PATH/unit/test_YOLOv5TRT.cpp \
    ../../ZeroMQ/Publisher.cpp \
    ../../ZeroMQ/Subscriber.cpp \
    sources/PeripheralController.cpp \
    sources/inference/CameraStreamer.cpp \
    sources/inference/TensorRTInferencer.cpp \
    sources/inference/LanePostProcessor.cpp \
    sources/inference/LaneCurveFitter.cpp \
    sources/objectDetection/LabelManager.cpp \
    sources/objectDetection/YOLOv5TRT.cpp

HEADERS += \
    $$TESTS_PATH/mocks/MockPeripheralController.hpp \
    $$TESTS_PATH/mocks/MockInferencer.hpp \
    ../../ZeroMQ/Publisher.hpp \
    ../../ZeroMQ/Subscriber.hpp \
    includes/inference/CameraStreamer.hpp \
    includes/inference/TensorRTInferencer.hpp \
    includes/inference/LanePostProcessor.hpp \
    includes/inference/LaneCurveFitter.hpp \
    includes/inference/IInferencer.hpp \
    includes/objectDetection/LabelManager.hpp \
    includes/objectDetection/YOLOv5TRT.hpp

# Library paths
LIBS += -L$${JETSON_SYSROOT}/usr/local/lib
LIBS += -L$${JETSON_SYSROOT}/usr/local/cuda-10.2/targets/aarch64-linux/lib
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/openblas
LIBS += -L$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/7
LIBS += -L/usr/local/lib  # For GLEW/GLFW

# GTest and GMock
GMOCK_LIBDIR = $${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
LIBS += -L$${GMOCK_LIBDIR} \
        -lgmock_main -lgtest_main -lgmock -lgtest -lpthread -lzmq

# TensorRT, CUDA, OpenCV
LIBS += -lcudart -lnvinfer
LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
LIBS += -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudacodec
LIBS += -lcublasLt -lnvmedia -lnvdla_compiler

# GStreamer libraries
LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

# OpenGL, GLEW, GLFW
LIBS += -lGLEW -lglfw -lGL

LIBS += -L$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/atlas  # Add ATLAS BLAS/LAPACK path
LIBS += -llapack -lcblas -lblas -ltbb

# Linker flags for rpath and static stdlib
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/local/lib
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/lib/aarch64-linux-gnu
QMAKE_LFLAGS += -Wl,-rpath-link,$${JETSON_SYSROOT}/usr/lib/gcc/aarch64-linux-gnu/7
QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/aarch64-linux-gnu
QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/gcc/aarch64-linux-gnu/7
QMAKE_LFLAGS += -Wl,-rpath,/usr/local/qt5.15/lib
