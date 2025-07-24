#include "Publisher.hpp"

std::unordered_map<int, Publisher*> Publisher::instances;

Publisher::Publisher(int port) : context(1), publisher(context, ZMQ_PUB), joytstick_value(true), running(false) {
	boundAddress = "tcp://*:" + std::to_string(port);
	publisher.bind(boundAddress);  // Dynamic port binding
}

Publisher* Publisher::instance(int port) {
	if (instances.find(port) == instances.end()) {
		instances[port] = new Publisher(port);
	}
	return instances[port];
}

void Publisher::publish(const std::string& topic, const std::string& message) {
	// std::cout << "[Publisher] Full message: " << topic << " " << message << std::endl;

	std::string full_message = topic + " " + message;
	zmq::message_t zmq_message(full_message.begin(), full_message.end());

	publisher.send(zmq_message);  // Send the message
}

void Publisher::setJoystickStatus(bool new_joytstick_value) {
	// std::cout << "[Publisher] Publishing joystick_value: " << (joytstick_value ? "true" : "false") << std::endl;

	std::lock_guard<std::mutex> lock(joystick_mtx);  // Ensure thread safety
	if (new_joytstick_value != joytstick_value) {
		joytstick_value = new_joytstick_value;
		std::string bool_str = joytstick_value ? "true" : "false";
		publish("joystick_value", bool_str);  // Publish a new status message
	}
}

void Publisher::publishInferenceFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image) {
	std::lock_guard<std::mutex> lock(frame_mtx);  // Ensure thread safety
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

		// Build single message: "topic " + raw image bytes
		std::string header = topic + " ";
		std::vector<uchar> messageData;
		messageData.reserve(header.size() + encoded.size());
		messageData.insert(messageData.end(), header.begin(), header.end());
		messageData.insert(messageData.end(), encoded.begin(), encoded.end());

		zmq::message_t zmq_message(messageData.data(), messageData.size());
		publisher.send(zmq_message);

		//std::cout << "[Publisher] Sent image as single-part message. Size: " << messageData.size() << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "[Publisher] Failed to publish image: " << e.what() << std::endl;
	}
}

void Publisher::publishPolyfittingResult(const std::string& topic, const LaneCurveFitter::CenterlineResult polyfitting_result) {
    std::lock_guard<std::mutex> lock(frame_mtx); // Ensure thread safety
    
    try {
        // Create JSON-like string representation of the result
        std::ostringstream json_stream;
        json_stream << "{";
        
        // Add validity flag
        json_stream << "\"valid\":" << (polyfitting_result.valid ? "true" : "false") << ",";
        
        // Add blended points
        json_stream << "\"blended\":[";
        for (size_t i = 0; i < polyfitting_result.blended.size(); ++i) {
            json_stream << "{\"x\":" << polyfitting_result.blended[i].x 
                       << ",\"y\":" << polyfitting_result.blended[i].y << "}";
            if (i < polyfitting_result.blended.size() - 1) json_stream << ",";
        }
        json_stream << "],";
        
        // Add midpoint points
        json_stream << "\"midpoint\":[";
        for (size_t i = 0; i < polyfitting_result.midpoint.size(); ++i) {
            json_stream << "{\"x\":" << polyfitting_result.midpoint[i].x 
                       << ",\"y\":" << polyfitting_result.midpoint[i].y << "}";
            if (i < polyfitting_result.midpoint.size() - 1) json_stream << ",";
        }
        json_stream << "],";
        
        // Add straight points
        json_stream << "\"straight\":[";
        for (size_t i = 0; i < polyfitting_result.straight.size(); ++i) {
            json_stream << "{\"x\":" << polyfitting_result.straight[i].x 
                       << ",\"y\":" << polyfitting_result.straight[i].y << "}";
            if (i < polyfitting_result.straight.size() - 1) json_stream << ",";
        }
        json_stream << "],";
        
        // Add lanes
        json_stream << "\"lanes\":[";
        for (size_t i = 0; i < polyfitting_result.lanes.size(); ++i) {
            json_stream << "{";
            
            // Add centroids for this lane
            json_stream << "\"centroids\":[";
            for (size_t j = 0; j < polyfitting_result.lanes[i].centroids.size(); ++j) {
                json_stream << "{\"x\":" << polyfitting_result.lanes[i].centroids[j].x 
                           << ",\"y\":" << polyfitting_result.lanes[i].centroids[j].y << "}";
                if (j < polyfitting_result.lanes[i].centroids.size() - 1) json_stream << ",";
            }
            json_stream << "],";
            
            // Add curve points for this lane
            json_stream << "\"curve\":[";
            for (size_t j = 0; j < polyfitting_result.lanes[i].curve.size(); ++j) {
                json_stream << "{\"x\":" << polyfitting_result.lanes[i].curve[j].x 
                           << ",\"y\":" << polyfitting_result.lanes[i].curve[j].y << "}";
                if (j < polyfitting_result.lanes[i].curve.size() - 1) json_stream << ",";
            }
            json_stream << "]";
            
            json_stream << "}";
            if (i < polyfitting_result.lanes.size() - 1) json_stream << ",";
        }
        json_stream << "]";
        
        json_stream << "}";
        
        // Get the JSON string
        std::string json_data = json_stream.str();
        
        // Use the existing publish method to send the data
        publish(topic, json_data);
        
    } catch (const std::exception& e) {
        std::cerr << "[Publisher] Failed to publish polyfitting result: " << e.what() << std::endl;
    }
}
