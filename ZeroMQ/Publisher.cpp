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

	publisher.send(zmq_message);
}

void Publisher::publishImageData(const std::string& topic, const std::vector<unsigned char>& data) {
	std::lock_guard<std::mutex> lock(image_mtx);
	if (!image_data.empty()) {
		// Combine the topic and data into a full message (topic + separator + data)
		std::vector<unsigned char> full_message;
		full_message.insert(full_message.end(), topic.begin(), topic.end());
		full_message.push_back(' ');
		full_message.insert(full_message.end(), data.begin(), data.end());  // Append image data

		zmq::message_t zmq_message(full_message.data(), full_message.size());
		publisher.send(zmq_message);
	}
}

void Publisher::setJoystickStatus(bool new_joytstick_value) {
	std::lock_guard<std::mutex> lock(joystick_mtx);
	if (new_joytstick_value != joytstick_value) {
		joytstick_value = new_joytstick_value;
		std::string bool_str = joytstick_value ? "true" : "false";
		publish("joystick_value", bool_str);
	}
}

void Publisher::setImageData(const std::vector<unsigned char>& new_image_data) {
	std::lock_guard<std::mutex> lock(image_mtx);  // Ensure thread safety
	image_data = new_image_data;
	publish("image_data", std::string(image_data.begin(), image_data.end()));
}
