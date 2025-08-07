QT       += core testlib
CONFIG   += c++17
TARGET   = car-controls-tests

CONFIG += debug
QMAKE_CXXFLAGS += -fprofile-arcs -ftest-coverage -O0
QMAKE_LFLAGS   += -fprofile-arcs -ftest-coverage

# Include Paths
INCLUDEPATH += \
    $$PWD/includes \
    $$PWD/tests/mocks \
    $$PWD/sources \
    /usr/local/include/opencv4 \
    /usr/include/opencv4 \
    /usr/local/include \
    /usr/include/eigen3 \
    /usr/include/gstreamer-1.0 \
    /usr/include/glib-2.0 \

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
LIBS += -L/usr/local/lib
LIBS += -L/usr/lib/x86_64-linux-gnu
LIBS += -L/usr/local/cuda/lib64

# GTest and GMock
LIBS += -lgmock_main -lgtest_main -lgmock -lgtest -lpthread -lzmq

# TensorRT and CUDA
LIBS += -lnvinfer -lcudart -lcublas -lcublasLt

# OpenCV + CUDA
LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui -lopencv_calib3d
LIBS += -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudaimgproc

# GStreamer
LIBS += -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

# OpenGL, GLEW, GLFW
LIBS += -lGLEW -lglfw -lGL

# BLAS/LAPACK
LIBS += -ltbb -llapack -lblas
