#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <zmq.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>

class Publisher {
private:
	zmq::context_t context;
	zmq::socket_t publisher;

	bool joytstick_value;
	std::mutex joystick_mtx;

	std::vector<unsigned char> image_data;
	std::mutex image_mtx;
	bool running;

public:
	Publisher();
	~Publisher();

	void publish(const std::string& topic, const std::string& message);
	void publishImageData(const std::string& topic, const std::vector<unsigned char>& data);
	void setJoystickStatus(bool new_joytstick_value);
	void setImageData(const std::vector<unsigned char>& new_image_data);
};

#endif // PUBLISHER_HPP
