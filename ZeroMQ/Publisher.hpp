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
	std::string boundAddress;
	bool running;

public:
	Publisher(int port);
	~Publisher();

	void publish(const std::string& topic, const std::string& message);
	void setJoystickStatus(bool new_joytstick_value);
	void publishFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image);
};

#endif // PUBLISHER_HPP
