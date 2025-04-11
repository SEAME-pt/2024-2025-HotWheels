#ifndef CAMERASTREAMER_HPP
#define CAMERASTREAMER_HPP

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include "TensorRTInferencer.hpp"

class CameraStreamer : public QObject {
	Q_OBJECT

public:
	explicit CameraStreamer(double scale = 0.5, const std::string& win_name = "CSI Camera", QObject *parent = nullptr);
	~CameraStreamer();

	void start();

private:
	cv::VideoCapture cap;
	double scale_factor;
	std::string window_name;

	TensorRTInferencer *inferencer;
};

#endif // CAMERASTREAMER_HPP
