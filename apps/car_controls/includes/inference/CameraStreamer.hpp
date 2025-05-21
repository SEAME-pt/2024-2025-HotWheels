#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <nvbuf_utils.h>
#include <NvEglRenderer.h>
#include <NvVideoCapture.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>

#include "TensorRTInferencer.hpp"
#include "../../../ZeroMQ/Subscriber.hpp"
#include "../../../ZeroMQ/Publisher.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

class CameraStreamer {
public:
	CameraStreamer(std::shared_ptr<TensorRTInferencer> inferencer, double scale = 0.5, const std::string& win_name = "CSI Camera", bool show_orig = false);
	~CameraStreamer();

	void initOpenGL();
	void initUndistortMaps();
	void uploadFrameToTexture(const cv::cuda::GpuMat& gpuFrame);
	void renderTexture();

	void start();
	void stop();

private:
	cv::VideoCapture cap;
	double scale_factor;
	std::string window_name;
	bool show_original;

	cv::cuda::GpuMat d_mapx, d_mapy;
	cudaGraphicsResource* cuda_resource;

	GLFWwindow* window;
	GLuint textureID;
	int window_width, window_height;

	bool m_running;
	std::shared_ptr<TensorRTInferencer> m_inferencer;

	Publisher *m_publisherObject;
};

#endif // CAMERA_STREAMER_HPP
