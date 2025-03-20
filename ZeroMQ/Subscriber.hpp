#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <zmq.hpp>
#include <iostream>
#include <string>

class Subscriber {
private:
	zmq::context_t context;
	zmq::socket_t subscriber;
    bool running;

public:
	Subscriber();
	~Subscriber();

	void subscribe(const std::string& topic);
	void listen();
	void connect(const std::string& address);
	void reconnect(const std::string& address);
	void stop();
};

#endif // SUBSCRIBER_HPP
