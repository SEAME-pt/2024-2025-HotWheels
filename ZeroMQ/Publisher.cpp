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
	std::cout << "[Publisher] Full message: " << topic << " " << message << std::endl;

	std::string full_message = topic + " " + message;
	zmq::message_t zmq_message(full_message.begin(), full_message.end());

	publisher.send(zmq_message);  // Send the message
}

void Publisher::setJoystickStatus(bool new_joytstick_value) {
	std::cout << "[Publisher] Publishing joystick_value: " << (joytstick_value ? "true" : "false") << std::endl;

	std::lock_guard<std::mutex> lock(joystick_mtx);  // Ensure thread safety
	if (new_joytstick_value != joytstick_value) {
		joytstick_value = new_joytstick_value;
		std::string bool_str = joytstick_value ? "true" : "false";
		publish("joystick_value", bool_str);  // Publish a new status message
	}
}

void Publisher::publishFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image) {
	try {
		// Download GPU image to CPU
		cv::Mat cpu_image;
		gpu_image.download(cpu_image);

		if (cpu_image.empty()) {
			std::cerr << "[Publisher] Skipped: empty CPU image." << std::endl;
			return;
		}

		// Encode to JPEG
		std::vector<uchar> encoded;
		if (!cv::imencode(".jpg", cpu_image, encoded)) {
			std::cerr << "[Publisher] Encoding failed." << std::endl;
			return;
		}

		// Construct message with deep copy (safe for legacy API)
		zmq::message_t topic_msg(topic.size());
		memcpy(topic_msg.data(), topic.data(), topic.size());

		zmq::message_t image_msg(encoded.size());
		memcpy(image_msg.data(), encoded.data(), encoded.size());

		// Send multipart message
		publisher.send(&topic_msg, ZMQ_SNDMORE);
		publisher.send(&image_msg, 0);

		std::cout << "[Publisher] Image sent. Topic: " << topic << ", Size: " << encoded.size() << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "[Publisher] Failed to publish image: " << e.what() << std::endl;
	}
}
