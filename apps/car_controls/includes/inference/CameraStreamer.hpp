#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>

#include "TensorRTInferencer.hpp"
#include "../../../ZeroMQ/Subscriber.hpp"
#include "../../../ZeroMQ/Publisher.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "IInferencer.hpp"

class CameraStreamer {
public:
	CameraStreamer(std::shared_ptr<IInferencer> inferencer, double scale = 0.5, const std::string& win_name = "CSI Camera", bool show_orig = false);
	~CameraStreamer();

	void initUndistortMaps();

	void start();
	void stop();

private:
	cv::VideoCapture cap;
	double scale_factor;
	std::string window_name;
	bool show_original;

	cv::cuda::GpuMat d_mapx, d_mapy;
	cudaGraphicsResource* cuda_resource;

	bool m_running;
	std::shared_ptr<IInferencer> m_inferencer;

	Publisher *m_publisherObject;
};

#endif // CAMERA_STREAMER_HPP
