#include "AutomaticMode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

AutomaticMode::AutomaticMode (EngineController *engineController, QObject *parent)
    : QObject (parent), m_curveFitter (nullptr),
      m_engineController (engineController), m_polyfittingSubscriber (nullptr), m_automaticMode (false), m_objectDetectionSubscriber (nullptr) {

	m_curveFitter = new LaneCurveFitter ();

    // std::cout << "[AutomaticMode] Initialized" << std::endl;
}

AutomaticMode::~AutomaticMode (void) {
	stopAutomaticControl ();

	if (m_curveFitter) {
		delete m_curveFitter;
		m_curveFitter = nullptr;
	}
}


void AutomaticMode::startAutomaticControl () {
	if (m_automaticMode == true) return;

    if (!m_polyfittingSubscriber) {
        m_polyfittingSubscriber = new Subscriber();
    }

	m_polyfittingSubscriber->connect(POLYFITTING_PORT);
    m_polyfittingSubscriber->subscribe(POLYFITTING_TOPIC);

    if (!m_objectDetectionSubscriber) {
        m_objectDetectionSubscriber = new Subscriber();
    }

    m_objectDetectionSubscriber->connect(OBJECT_PORT);
    m_objectDetectionSubscriber->subscribe(OBJECT_TOPIC);
    
	m_automaticControlThread = QThread::create ([this] () { automaticControlLoop (); });
	m_automaticControlThread->start ();

    // std::cout << "[AutomaticMode] Automatic control started" << std::endl;
    m_automaticMode = true;
}

void AutomaticMode::stopAutomaticControl () {
	if (m_automaticMode == false) return;

    m_automaticMode = false;
    
    if (m_polyfittingSubscriber) {
        delete m_polyfittingSubscriber;
        m_polyfittingSubscriber = nullptr;
    }

    if (m_objectDetectionSubscriber) {
        delete m_objectDetectionSubscriber;
        m_objectDetectionSubscriber = nullptr;
    }

	if (m_automaticControlThread) {
        m_automaticControlThread->quit ();
		if (!m_automaticControlThread->wait (2000)) {
			m_automaticControlThread->terminate ();
			m_automaticControlThread->wait (1000);
		}
		delete m_automaticControlThread;
		m_automaticControlThread = nullptr;
	}
    
	if (m_engineController) {
        m_engineController->set_speed (0);
		m_engineController->set_steering (0);
	}
}

void AutomaticMode::automaticControlLoop () {
    auto last_control_time = std::chrono::steady_clock::now();
    auto last_slowdown_time = std::chrono::steady_clock::now();

	while (m_automaticMode == true) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - last_control_time).count();
        auto elapsed_slowdown = std::chrono::duration<double>(current_time - last_slowdown_time).count();

        if (elapsed >= COMMAND_DELAY_S) {
            try {
                auto newShouldSlowDown = getShouldSlowDown ();
                if (m_shouldSlowDown && newShouldSlowDown == false && elapsed_slowdown >= SLOW_DOWN_DURATION_S) {
                    m_shouldSlowDown = false;
                } else if (newShouldSlowDown == true) {
                    m_shouldSlowDown = true;
                    last_slowdown_time = current_time;
                }
                auto polyfitting_result = getPolyfittingResult ();
                if (polyfitting_result.valid) {
                    ControlCommand command = calculateSteering (polyfitting_result);
                    // std::cout << "[AutomaticMode] Applying controls: Throttle = "
                    //           << command.throttle << ", Steering = " << command.steer << std::endl;
                    applyControls (command);
                }
            } catch (const std::exception &e) {
                if (m_engineController) {
                    m_engineController->set_speed (0); // Safety
                }
            }
            last_control_time = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

void AutomaticMode::applyControls (const ControlCommand &control) {
	if (!m_engineController) return;

	// Apply to hardware
	m_engineController->set_speed (-control.throttle); // Inverted for motor cross-connection
	m_engineController->set_steering (control.steer);
}

ControlCommand AutomaticMode::calculateSteering(const LaneCurveFitter::CenterlineResult &centerline_result) {
    ControlCommand controlCommand;
	    
    try {
        auto angle = computeDirectionAngle(centerline_result.blended);

        controlCommand.steer = angle;
		
        if (this->m_shouldSlowDown) {
            controlCommand.throttle = TURN_SPEED_THROTTLE;
            std::cout << "[AutomaticMode] Slowing down due to detected object" << std::endl;
        } else if (angle > TURN_ANGLE_THRESHOLD || angle < -TURN_ANGLE_THRESHOLD) {
    		controlCommand.throttle = STRAIGHT_SPEED_THROTTLE;
            std::cout << "[AutomaticMode] Slowing down for turn, angle: " << angle << std::endl;
		} else {
			controlCommand.throttle = STRAIGHT_SPEED_THROTTLE;
		}
    } catch (const std::exception &e) {
        controlCommand.steer = 0;
        return controlCommand;
    }
    
    return controlCommand;
}

int AutomaticMode::computeDirectionAngle(const std::vector<Point2D>& centerline)
{
    if (centerline.size() < 6) return 0.0;

    size_t N = centerline.size();
    size_t startIdx = N * LOOK_AHEAD_START;
    size_t endIdx = N * LOOK_AHEAD_END;

    const auto& p0 = centerline[startIdx];
    const auto& p1 = centerline[endIdx];

    double dx = p1.x - p0.x;
    double dy = p0.y - p1.y;

    if (std::abs(dy) < 1e-3) dy = 1e-3;

    double angle_rad = std::atan2(dx, dy);
    double angle_deg = (angle_rad * 180.0 / CV_PI);

	if (angle_deg < 0) {
		angle_deg = angle_deg * LEFT_STEERING_SCALE;
	} else {
		angle_deg = angle_deg * RIGHT_STEERING_SCALE;
	}

    if (angle_deg > MAX_STEERING_ANGLE) {
        angle_deg = MAX_STEERING_ANGLE;
    } else if (angle_deg < -MAX_STEERING_ANGLE) {
        angle_deg = -MAX_STEERING_ANGLE;
    }



    return static_cast<int> (angle_deg);
}



LaneCurveFitter::CenterlineResult AutomaticMode::extractJsonData(std::string data) {
    LaneCurveFitter::CenterlineResult result;
    
    try {
        // Remove leading/trailing whitespace and topic prefix if present
        size_t start = data.find('{');
        if (start == std::string::npos) {
            std::cerr << "[AutomaticMode] Invalid JSON data received: " << data << std::endl;
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
			// std::cout << "[AutomaticMode] Polyfitting result valid: " << result.valid << std::endl;
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
        std::cerr << "[AutomaticMode] Error extracting JSON data: " << e.what() << std::endl;
        result.valid = false;
    }
    
    return result;
}

std::vector<Point2D> AutomaticMode::parsePointArray(const std::string& json_data, const std::string& field_name) {
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

std::vector<LaneCurveFitter::LaneCurve> AutomaticMode::parseLaneArray(const std::string& json_data) {
    std::vector<LaneCurveFitter::LaneCurve> lanes;
    
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
                
                LaneCurveFitter::LaneCurve lane;
                
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

bool AutomaticMode::getShouldSlowDown() const {
    if (!m_objectDetectionSubscriber) {
        std::cerr << "[AutomaticMode] Object detection subscriber not initialized!" << std::endl;
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
                            std::cout << "[AutomaticMode] Detected object " << object << ", slowing down" << std::endl;
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
        std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
        return {};
    }
}

LaneCurveFitter::CenterlineResult AutomaticMode::getPolyfittingResult() {
    if (!m_polyfittingSubscriber) {
        std::cerr << "[AutomaticMode] Polyfitting subscriber not initialized!" << std::endl;
        return {};
    }

    try {
        zmq::pollitem_t items[] = {
            { static_cast<void*>(m_polyfittingSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
        };
        
        LaneCurveFitter::CenterlineResult latestResult;
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
        
        // if (messagesProcessed > 1) {
        //     std::cout << "[Subscriber] Drained " << messagesProcessed << " messages, using latest" << std::endl;
        // }
        
        if (foundValidMessage) {
            return latestResult;
        }
        
    } catch (const zmq::error_t& e) {
        std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
        return {};
    }

    // No valid message found
    return {};
}

