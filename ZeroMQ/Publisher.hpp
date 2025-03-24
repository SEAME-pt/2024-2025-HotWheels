#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <zmq.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>

class Publisher {
private:
	zmq::context_t context;
	zmq::socket_t publisher;
	bool joytstick_value;
	std::mutex joystick_mtx;
	bool running;

public:
	Publisher();
	~Publisher();

	void publish(const std::string& topic, const std::string& message);
	void setJoystickStatus(bool new_joytstick_value);
};

#endif // PUBLISHER_HPP
