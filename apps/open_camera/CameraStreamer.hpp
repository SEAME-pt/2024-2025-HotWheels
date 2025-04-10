#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include "TensorRTInferencer.hpp"

class CameraStreamer {
private:
	cv::VideoCapture cap;
	double scale_factor;
	std::string window_name;

	TensorRTInferencer *inferencer;

public:
	CameraStreamer(double scale = 0.5, const std::string& win_name = "CSI Camera",);
	~CameraStreamer();

	void start();
};

#endif // CAMERA_STREAMER_HPP
