#include "ControlDataHandler.hpp"
#include <iostream>
#include <stdexcept>

ControlDataHandler::ControlDataHandler()
    : m_polyfittingSubscriber(nullptr),
    m_objectDetectionSubscriber(nullptr),
    m_carSpeedSubscriber(nullptr),
    m_ultrasoundSubscriber(nullptr),
    m_ultrasoundThread(nullptr),
    m_ultraRunning(true) {
}

ControlDataHandler::~ControlDataHandler() {
    cleanupSubscribers();
}

void ControlDataHandler::initializeSubscribers() {
    // Initialize polyfitting subscriber
    if (!m_polyfittingSubscriber) {
        m_polyfittingSubscriber = new Subscriber();
    }
    m_polyfittingSubscriber->connect(getZeroMQAddress(POLYFITTING_PORT));
    m_polyfittingSubscriber->subscribe(POLYFITTING_TOPIC);

    // Initialize object detection subscriber
    if (!m_objectDetectionSubscriber) {
        m_objectDetectionSubscriber = new Subscriber();
    }
    m_objectDetectionSubscriber->connect(getZeroMQAddress(OBJECT_PORT));
    m_objectDetectionSubscriber->subscribe(OBJECT_TOPIC);

    // Initialize car speed subscriber
    if (!m_carSpeedSubscriber) {
        m_carSpeedSubscriber = new Subscriber();
    }
    m_carSpeedSubscriber->connect(getZeroMQAddress(CAR_SPEED_PORT));
    m_carSpeedSubscriber->subscribe(CAR_SPEED_TOPIC);

    // Initialize ultrasound subscriber thread
	m_ultrasoundSubscriber = new Subscriber();
	m_ultrasoundThread = QThread::create([this]()
									{
		m_ultrasoundSubscriber->connect(getZeroMQAddress(ULTRASOUND_PORT));
		m_ultrasoundSubscriber->subscribe(ULTRASOUND_TOPIC);
		while (m_ultraRunning) {
			try {
				zmq::pollitem_t items[] = {
					{ static_cast<void*>(m_ultrasoundSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
				};

				// Wait up to 100ms for a message
				zmq::poll(items, 1, 100);

				if (items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if (!m_ultrasoundSubscriber->getSocket().recv(&message, 0)) {
						continue;  // failed to receive
					}

					std::string received_msg(static_cast<char*>(message.data()), message.size());

					if (received_msg.find("ultrasound") == 0) {
						std::string value = received_msg.substr(std::string("ultrasound ").length());
						double cm = std::stod(value);
						double meters = cm / 100.0;
						m_latestUltraMeters.store(meters, std::memory_order_relaxed);
						std::cout << "Value received ultrasound: " << m_latestUltraMeters << " m\n";
					}
				}
			} catch (const zmq::error_t& e) {
				std::cerr << "[Subscriber] Ultrasound ZMQ error: " << e.what() << std::endl;
				break;  // exit safely if socket is closed
			}
		}
	});
	m_ultrasoundThread->start();
}

void ControlDataHandler::cleanupSubscribers() {
    if (m_polyfittingSubscriber) {
        delete m_polyfittingSubscriber;
        m_polyfittingSubscriber = nullptr;
    }

    if (m_objectDetectionSubscriber) {
        delete m_objectDetectionSubscriber;
        m_objectDetectionSubscriber = nullptr;
    }

    if (m_carSpeedSubscriber) {
        delete m_carSpeedSubscriber;
        m_carSpeedSubscriber = nullptr;
    }

    m_ultraRunning = false;
    if (m_ultrasoundThread) {
        m_ultrasoundThread->wait();
        delete m_ultrasoundThread;
        m_ultrasoundThread = nullptr;
    }

    if (m_ultrasoundSubscriber) {
        delete m_ultrasoundSubscriber;
        m_ultrasoundSubscriber = nullptr;
    }
}

float ControlDataHandler::getCarSpeed() const {
    if (!m_carSpeedSubscriber) {
        std::cerr << "[ControlDataHandler] Car speed subscriber not initialized!" << std::endl;
        return 0.0f;
    }

    try {
        zmq::pollitem_t items[] = {
            { static_cast<void*>(m_carSpeedSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
        };

        float latestSpeed = 0.0f;
        bool foundValidMessage = false;

        // Drain the queue to get the most recent message
        while (true) {
            // Poll with short timeout to check for available messages
            zmq::poll(items, 1, 10); // 10ms timeout

            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t message;
                if (!m_carSpeedSubscriber->getSocket().recv(&message, ZMQ_DONTWAIT)) {
                    break; // No more messages
                }

                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = CAR_SPEED_TOPIC;

                if (received_msg.find(topic) == 0) {
                    // Extract speed value from the message (skip topic and space)
                    std::string speed_str = received_msg.substr(topic.size() + 1);

                    try {
                        if (speed_str.find('.') != std::string::npos) {
                            speed_str.replace(speed_str.find('.'), 1, ",");
                        }
                        latestSpeed = std::stof(speed_str);
                        foundValidMessage = true;
                    } catch (const std::exception& e) {
                        std::cerr << "[ControlDataHandler] Error parsing speed value '" << speed_str << "': " << e.what() << std::endl;
                    }
                }
            } else {
                break; // No more messages available
            }
        }

        if (foundValidMessage) {
            // std::cout << "[ControlDataHandler] Final parsed speed: " << latestSpeed << std::endl;
            return latestSpeed;
        }

    } catch (const zmq::error_t& e) {
        std::cerr << "[ControlDataHandler] ZMQ error (getCarSpeed): " << e.what() << std::endl;
    }

    return 0.0f; // Default speed if no message is received
}

CenterlineResult ControlDataHandler::getPolyfittingResult() {
    if (!m_polyfittingSubscriber) {
        std::cerr << "[ControlDataHandler] Polyfitting subscriber not initialized!" << std::endl;
        return {};
    }

    try {
        zmq::pollitem_t items[] = {
            { static_cast<void*>(m_polyfittingSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
        };

        CenterlineResult latestResult;
        bool foundValidMessage = false;
        int messagesProcessed = 0;

        // Drain the queue to get the most recent message
        while (true) {
            // Poll with short timeout to check for available messages
            zmq::poll(items, 1, 10); // 10ms timeout

            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t message;
                if (!m_polyfittingSubscriber->getSocket().recv(&message, ZMQ_DONTWAIT)) {
                    break; // No more messages
                }

                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = POLYFITTING_TOPIC;

                if (received_msg.find(topic) == 0) {
                    auto extractedResult = extractJsonData(received_msg.substr(topic.size()));
                    if (extractedResult.valid) {
                        latestResult = extractedResult;
                        foundValidMessage = true;
                    }
                    messagesProcessed++;
                }
            } else {
                break; // No more messages available
            }
        }

        if (foundValidMessage) {
            return latestResult;
        }

    } catch (const zmq::error_t& e) {
        std::cerr << "[ControlDataHandler] ZMQ error (getPolytfittingReuslt): " << e.what() << std::endl;
        return CenterlineResult();
    }

    // No valid message found
    return CenterlineResult();
}

bool ControlDataHandler::getShouldSlowDown() const {
    if (!m_objectDetectionSubscriber) {
        std::cerr << "[ControlDataHandler] Object detection subscriber not initialized!" << std::endl;
        return false;
    }

    try {
        zmq::pollitem_t items[] = {
            { static_cast<void*>(m_objectDetectionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
        };

        int messagesProcessed = 0;
        bool shouldSlowDown = false;

        // Drain the queue to get the most recent message
        while (true) {
            // Poll with short timeout to check for available messages
            zmq::poll(items, 1, 10); // 10ms timeout

            if (items[0].revents & ZMQ_POLLIN) {
                zmq::message_t message;
                if (!m_objectDetectionSubscriber->getSocket().recv(&message, ZMQ_DONTWAIT)) {
                    break; // No more messages
                }

                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = OBJECT_TOPIC;

                if (received_msg.find(topic) == 0) {
                    for (const auto& object : SLOW_DOWN_OBJECTS) {
                        if (received_msg.find(object) != std::string::npos) {
                            shouldSlowDown = true;
                            std::cout << "[ControlDataHandler] Detected object " << object << ", slowing down" << std::endl;
                            break;
                        }
                    }
                    messagesProcessed++;
                }
            } else {
                break; // No more messages available
            }
        }

        return shouldSlowDown;

    } catch (const zmq::error_t& e) {
        std::cerr << "[ControlDataHandler] ZMQ error (getShouldSlowDown): " << e.what() << std::endl;
        return false;
    }
}

CenterlineResult ControlDataHandler::extractJsonData(std::string data) {
    CenterlineResult result;

    try {
        // Remove leading/trailing whitespace and topic prefix if present
        size_t start = data.find('{');
        if (start == std::string::npos) {
            std::cerr << "[ControlDataHandler] Invalid JSON data received: " << data << std::endl;
            return result;
        }

        std::string json_data = data.substr(start);

        // Parse valid field
        size_t valid_pos = json_data.find("\"valid\":");
        if (valid_pos != std::string::npos) {
            size_t value_start = json_data.find(":", valid_pos) + 1;
            size_t value_end = json_data.find(",", value_start);
            std::string valid_str = json_data.substr(value_start, value_end - value_start);
            result.valid = (valid_str.find("true") != std::string::npos);
        }

        // Parse blended points
        result.blended = parsePointArray(json_data, "\"blended\":");

        // Parse midpoint points
        result.midpoint = parsePointArray(json_data, "\"midpoint\":");

        // Parse straight points
        result.straight = parsePointArray(json_data, "\"straight\":");

        // Parse lanes
        result.lanes = parseLaneArray(json_data);

    } catch (const std::exception& e) {
        std::cerr << "[ControlDataHandler] Error extracting JSON data: " << e.what() << std::endl;
        result.valid = false;
    }

    return result;
}

std::vector<Point2D> ControlDataHandler::parsePointArray(const std::string& json_data, const std::string& field_name) {
    std::vector<Point2D> points;

    size_t field_pos = json_data.find(field_name);
    if (field_pos == std::string::npos) {
        return points;
    }

    size_t array_start = json_data.find("[", field_pos);
    if (array_start == std::string::npos) {
        return points;
    }

    size_t array_end = json_data.find("]", array_start);
    if (array_end == std::string::npos) {
        return points;
    }

    std::string array_content = json_data.substr(array_start + 1, array_end - array_start - 1);

    // Parse individual point objects
    size_t pos = 0;
    while (pos < array_content.length()) {
        size_t obj_start = array_content.find("{", pos);
        if (obj_start == std::string::npos) break;

        size_t obj_end = array_content.find("}", obj_start);
        if (obj_end == std::string::npos) break;

        std::string point_obj = array_content.substr(obj_start, obj_end - obj_start + 1);

        // Parse x coordinate
        size_t x_pos = point_obj.find("\"x\":");
        if (x_pos != std::string::npos) {
            size_t x_start = point_obj.find(":", x_pos) + 1;
            size_t x_end = point_obj.find(",", x_start);
            if (x_end == std::string::npos) x_end = point_obj.find("}", x_start);

            // Parse y coordinate
            size_t y_pos = point_obj.find("\"y\":");
            if (y_pos != std::string::npos) {
                size_t y_start = point_obj.find(":", y_pos) + 1;
                size_t y_end = point_obj.find("}", y_start);

                try {
                    double x = std::stod(point_obj.substr(x_start, x_end - x_start));
                    double y = std::stod(point_obj.substr(y_start, y_end - y_start));
                    points.push_back(Point2D{x, y});
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing point: " << e.what() << std::endl;
                }
            }
        }

        pos = obj_end + 1;
    }

    return points;
}

std::vector<LaneCurve> ControlDataHandler::parseLaneArray(const std::string& json_data) {
    std::vector<LaneCurve> lanes;

    size_t lanes_pos = json_data.find("\"lanes\":");
    if (lanes_pos == std::string::npos) {
        return lanes;
    }

    size_t array_start = json_data.find("[", lanes_pos);
    if (array_start == std::string::npos) {
        return lanes;
    }

    size_t array_end = json_data.rfind("]"); // Find the last closing bracket
    if (array_end == std::string::npos) {
        return lanes;
    }

    std::string array_content = json_data.substr(array_start + 1, array_end - array_start - 1);

    // Parse individual lane objects
    size_t pos = 0;
    int brace_count = 0;
    size_t lane_start = 0;

    for (size_t i = 0; i < array_content.length(); ++i) {
        if (array_content[i] == '{') {
            if (brace_count == 0) {
                lane_start = i;
            }
            brace_count++;
        } else if (array_content[i] == '}') {
            brace_count--;
            if (brace_count == 0) {
                std::string lane_obj = array_content.substr(lane_start, i - lane_start + 1);

                LaneCurve lane;

                // Parse centroids
                lane.centroids = parsePointArray(lane_obj, "\"centroids\":");

                // Parse curve points
                lane.curve = parsePointArray(lane_obj, "\"curve\":");

                lanes.push_back(lane);
            }
        }
    }

    return lanes;
}
