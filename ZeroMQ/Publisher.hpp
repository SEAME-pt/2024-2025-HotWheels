#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <zmq.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <opencv2/opencv.hpp>   // Main OpenCV functionality (includes other headers)
#include <opencv2/imgcodecs.hpp>  // For reading and writing images (imread, imencode)
#include <opencv2/highgui.hpp>    // For displaying images (imshow, waitKey)
#include <opencv2/imgproc.hpp> // For image processing (resize, etc.)

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

		void publishJoytstickStatus(const std::string& topic, const std::string& message);
		void publishImageData(const std::string& topic, const std::vector<unsigned char>& data);
		void setJoystickStatus(bool new_joytstick_value);
		void setImageData(const std::vector<unsigned char>& new_image_data);
		void sendImage(const std::string& image_path);
};

#endif // PUBLISHER_HPP
