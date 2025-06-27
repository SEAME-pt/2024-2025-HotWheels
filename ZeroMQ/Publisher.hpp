#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <zmq.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

class Publisher {
private:

zmq::context_t context;
zmq::socket_t publisher;
bool joytstick_value;
std::mutex joystick_mtx;
std::mutex frame_mtx;
std::string boundAddress;
bool running;

static std::unordered_map<int, std::shared_ptr<Publisher>> instances;

public:
	explicit Publisher(int port);
	//Publisher(int port);
	~Publisher();
	//static std::shared_ptr<Publisher> m_instance;
	static void destroyAll();

	// Singleton accessor
	static std::shared_ptr<Publisher> instance(int port);  // default port

	void publish(const std::string& topic, const std::string& message);
	void setJoystickStatus(bool new_joytstick_value);
	void publishInferenceFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image);
	//void publishCameraFrame(const std::string& topic, const cv::Mat& frame);
};

#endif // PUBLISHER_HPP
