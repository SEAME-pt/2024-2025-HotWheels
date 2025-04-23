#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "TensorRTInferencer.hpp"

class CameraStreamer {
private:
    cv::VideoCapture cap;
    double scale_factor;
    std::string window_name;
    cv::cuda::GpuMat d_mapx, d_mapy;  // Undistortion maps
    TensorRTInferencer& inferencer;
    bool show_original;

public:
    CameraStreamer(TensorRTInferencer& infer, double scale = 0.5,
                  const std::string& win_name = "CSI Camera", bool show_orig = false);
    ~CameraStreamer();
    void start();
    void initUndistortMaps();
};

#endif // CAMERA_STREAMER_HPP
