#include "Subscriber.hpp"
#include <iostream>
#include <chrono>
#include <thread>

Subscriber::Subscriber() : context(1), subscriber(context, ZMQ_SUB), running(false) {}

Subscriber::~Subscriber() {
	stop();  // Ensure that the subscriber stops when destroyed
}

void Subscriber::connect(const std::string& address) {
	bool connected = false;

	// Attempt to connect until successful
	while (!connected) {
		try {
			subscriber.connect(address);  // Attempt to connect to the publisher
			std::cout << "Subscriber connected to " << address << std::endl;
			connected = true; // Exit the loop once the connection is successful
		}
		catch (const zmq::error_t& e) {
			std::cout << "Connection failed, retrying in 1 second..." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));  // Wait before retrying
		}
	}
}

zmq::socket_t& Subscriber::getSocket() {
	return subscriber;
}

void Subscriber::subscribe(const std::string& topic) {
	// Subscribe to a topic only after successfully connecting
	subscriber.setsockopt(ZMQ_SUBSCRIBE, topic.c_str(), topic.size());
}

void Subscriber::stop() {
	running = false;
	//subscriber.close();  // Close the socket gracefully
}
