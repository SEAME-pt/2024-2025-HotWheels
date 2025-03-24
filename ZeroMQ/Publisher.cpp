#include "Publisher.hpp"

Publisher::Publisher() : context(1), publisher(context, ZMQ_PUB), joytstick_value(true), running(false) {
	publisher.bind("tcp://*:5555");  // Bind to the publisher socket
}

Publisher::~Publisher() {
	// Unbind the socket before cleanup
	publisher.unbind("tcp://*:5555");
}

void Publisher::publish(const std::string& topic, const std::string& message) {
	std::string full_message = topic + " " + message;
	zmq::message_t zmq_message(full_message.begin(), full_message.end());

	publisher.send(zmq_message);  // Send the message
}

void Publisher::setJoystickStatus(bool new_joytstick_value) {
	std::lock_guard<std::mutex> lock(joystick_mtx);  // Ensure thread safety
	if (new_joytstick_value != joytstick_value) {
		joytstick_value = new_joytstick_value;
		std::string bool_str = joytstick_value ? "true" : "false";
		publish("joystick_value", bool_str);  // Publish a new status message
	}
}
