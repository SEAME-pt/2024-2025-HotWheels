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

void Subscriber::subscribe(const std::string& topic) {
    // Subscribe to a topic only after successfully connecting
    subscriber.setsockopt(ZMQ_SUBSCRIBE, topic.c_str(), topic.size());
    std::cout << "Subscribed to topic: " << topic << std::endl;
}

void Subscriber::listen() {
    running = true;  // Mark the subscriber as running

    while (running) {
        try {
            zmq::message_t message;
            subscriber.recv(message, 0);

            std::string received_msg(static_cast<char*>(message.data()), message.size());
            std::cout << "Received: " << received_msg << std::endl;
        }
        catch (const zmq::error_t& e) {
            if (running) {  // If running is still true, handle reconnection
                std::cout << "Connection lost. Attempting to reconnect..." << std::endl;
                reconnect("tcp://localhost:5555");
            }
        }
    }

    std::cout << "Listener stopped." << std::endl;
}

void Subscriber::reconnect(const std::string& address) {
    bool connected = false;

    // Retry the connection until successful
    while (!connected && running) {
        try {
            std::cout << "Reconnecting to " << address << "..." << std::endl;
            subscriber.connect(address);  // Attempt to reconnect to the publisher
            std::cout << "Reconnected successfully." << std::endl;
            connected = true; // Exit the loop once the connection is successful
        }
        catch (const zmq::error_t& e) {
            if (!running) return;  // Exit if running is set to false
            std::cout << "Reconnection failed, retrying in 1 second..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));  // Wait before retrying
        }
    }
    if (connected) {
        listen();  // Recurse into listen after reconnecting
    }
}

void Subscriber::stop() {
    running = false;  // Set the running flag to false, causing the listen loop to stop
    subscriber.close();  // Close the socket gracefully
    std::cout << "Subscriber stopped." << std::endl;
}
