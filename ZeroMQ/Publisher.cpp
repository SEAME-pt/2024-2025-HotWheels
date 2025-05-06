#include "Publisher.hpp"

Publisher::Publisher(int port) : context(1), publisher(context, ZMQ_PUB), joytstick_value(true), running(false) {
	boundAddress = "tcp://*:" + std::to_string(port);
	publisher.bind(boundAddress);  // Dynamic port binding
}

Publisher::~Publisher() {
	try {
		publisher.unbind(boundAddress);  // Use stored address
		std::cout << "[Publisher] Unbound from " << boundAddress << std::endl;
	} catch (const zmq::error_t& e) {
		std::cerr << "[Publisher] Failed to unbind: " << e.what() << std::endl;
	}
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

void Publisher::publishFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image) {
	try {
		// Download from GPU to CPU
		cv::Mat cpu_image;
		gpu_image.download(cpu_image);  // Minimal CPU usage

		// Encode as JPEG
		std::vector<uchar> encoded;
		cv::imencode(".jpg", cpu_image, encoded);

		// Create ZeroMQ multipart message: [topic][JPEG image]
		zmq::message_t topic_msg(topic.data(), topic.size());
		zmq::message_t image_msg(encoded.data(), encoded.size());

		publisher.send(topic_msg, zmq::send_flags::sndmore);
		publisher.send(image_msg, zmq::send_flags::none);

	} catch (const std::exception& e) {
		std::cerr << "[Publisher] Failed to publish image: " << e.what() << std::endl;
	}
}
