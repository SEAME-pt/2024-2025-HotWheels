#include "AutonomousMode.hpp"
#include "Debugger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

AutonomousMode::AutonomousMode (EngineController *engineController, QObject *parent)
    : QObject (parent), m_mpcPlanner (nullptr), m_curveFitter (nullptr),
      m_engineController (engineController), m_inferenceSubscriber (nullptr){

	// Initialize MPC components
	initializeMPCComponents ();
}

AutonomousMode::~AutonomousMode (void) {
	stopAutonomousControl ();

	// Clean up MPC objects
	if (m_mpcPlanner) {
		delete m_mpcPlanner;
		m_mpcPlanner = nullptr;
	}
	if (m_curveFitter) {
		delete m_curveFitter;
		m_curveFitter = nullptr;
	}
}

void AutonomousMode::initializeMPCComponents () {
	// === Polyfitter for Lane Processing ===
	m_curveFitter = new LaneCurveFitter ();

	// === MPC Planner (lazy initialization) ===
	m_mpcPlanner = nullptr; // Will be created when autonomous mode starts

	m_inferenceSubscriber = new Subscriber();
	m_inferenceSubscriber->connect("tcp://localhost:5569");
    m_inferenceSubscriber->subscribe("polyfitting_result");

	INFO_LOG ("AutonomousMode", "MPC components initialized");
}

LaneCurveFitter::CenterlineResult AutonomousMode::extractJsonData(std::string data) {
    LaneCurveFitter::CenterlineResult result;
    
    try {
        // Remove leading/trailing whitespace and topic prefix if present
        size_t start = data.find('{');
        if (start == std::string::npos) {
            ERROR_LOG("AutonomousMode", "Invalid JSON format - no opening brace found");
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
        
        std::cout << "[AutonomousMode] Parsed CenterlineResult: valid=" << result.valid 
                  << ", blended=" << result.blended.size() 
                  << ", midpoint=" << result.midpoint.size() 
                  << ", straight=" << result.straight.size() 
                  << ", lanes=" << result.lanes.size() << std::endl;
        
    } catch (const std::exception& e) {
        ERROR_LOG("AutonomousMode", "Exception parsing JSON data: " + std::string(e.what()));
        result.valid = false;
    }
    
    return result;
}

// Helper method to parse an array of Point2D objects
std::vector<Point2D> AutonomousMode::parsePointArray(const std::string& json_data, const std::string& field_name) {
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
                    ERROR_LOG("AutonomousMode", "Error parsing point coordinates: " + std::string(e.what()));
                }
            }
        }
        
        pos = obj_end + 1;
    }
    
    return points;
}

// Helper method to parse the lanes array
std::vector<LaneCurveFitter::LaneCurve> AutonomousMode::parseLaneArray(const std::string& json_data) {
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



LaneCurveFitter::CenterlineResult AutonomousMode::getPolyfittingResult () {
	if (!m_inferenceSubscriber) {
		ERROR_LOG ("AutonomousMode", "Inference subscriber not initialized");
		return {};
	}

	try {
          zmq::pollitem_t items[] = {
            { static_cast<void*>(m_inferenceSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
          };
          zmq::poll(items, 1, 100);  // Timeout: 100 ms
          if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if (!m_inferenceSubscriber->getSocket().recv(&message, 0)) {
              ERROR_LOG ("AutonomousMode", "Failed to receive message from subscriber");
			  return {};
            }
            std::string received_msg(static_cast<char*>(message.data()), message.size());
            //std::cout << "[Subscriber] Raw message: " << received_msg.substr(0, 30) << "... (" << message.size() << " bytes)" << std::endl;
            const std::string topic = "polyfitting_result";;
            if (received_msg.find(topic) == 0) {
				auto extractedResult = extractJsonData(received_msg.substr(topic.size()));
					return extractedResult;
            }
          }
        } catch (const zmq::error_t& e) {
          std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
          return {};
        }

	ERROR_LOG ("AutonomousMode", "Failed to receive Polyfitting result");
	return {};
}


void AutonomousMode::startAutonomousControl () {
	if (m_autonomousMode.load ()) return;

	m_autonomousMode = true;

	// Create MPC planner if not already created
	if (!m_mpcPlanner) {
		m_mpcPlanner = new MPCPlanner ();
	}

	std::cout << "[SOFT START] Autonomous mode activated with gradual acceleration" << std::endl;

	m_autonomousControlThread = QThread::create ([this] () { autonomousControlLoop (); });
	m_autonomousControlThread->start ();
}

void AutonomousMode::stopAutonomousControl () {
	if (!m_autonomousMode.load ()) return;

	INFO_LOG ("AutonomousMode", "Stopping autonomous control...");
	m_autonomousMode = false;

	if (m_autonomousControlThread) {
		m_autonomousControlThread->quit ();
		if (!m_autonomousControlThread->wait (2000)) {
			WARNING_LOG ("AutonomousMode", "Autonomous thread did not finish gracefully");
			m_autonomousControlThread->terminate ();
			m_autonomousControlThread->wait (1000);
		}
		delete m_autonomousControlThread;
		m_autonomousControlThread = nullptr;
	}

	if (m_engineController) {
		m_engineController->set_speed (0);
		m_engineController->set_steering (0);
	}
	INFO_LOG ("AutonomousMode", "Autonomous control stopped successfully");
}

void AutonomousMode::autonomousControlLoop () {
	const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
	auto last_control_time = std::chrono::steady_clock::now ();

	INFO_LOG ("AutonomousMode", "Autonomous control loop started (MPC delegated)");

	while (m_autonomousMode.load ()) {
		// Emergency stop check
		if (m_emergencyStop.load ()) {
			if (m_engineController) {
				m_engineController->set_speed (0);
				m_engineController->set_steering (0);
			}
			INFO_LOG ("AutonomousMode", "Emergency stop is active - motors stopped");
			std::this_thread::sleep_for (std::chrono::milliseconds (50));
			continue;
		}

		// Control period timing
		auto now = std::chrono::steady_clock::now ();
		auto elapsed = std::chrono::duration<double> (now - last_control_time).count ();
		if (elapsed < CONTROL_PERIOD) {
			std::this_thread::sleep_for (std::chrono::microseconds (
			    static_cast<int> ((CONTROL_PERIOD - elapsed) * 1e6 * 0.8)));
			continue;
		}
		last_control_time = now;

		static int control_counter = 0;
		control_counter++;

		try {
			// Check emergency obstacles
			if (getCachedEmergencyStop ()) {
				if (m_engineController) {
					m_engineController->set_speed (0);
				}
				if (control_counter % 40 == 0) {
					INFO_LOG ("AutonomousMode", "Emergency stop activated!");
				}
				continue;
			}

			// auto inference_data = getInferenceFrameData ();
			auto polyfitting_result = getPolyfittingResult ();
			if (polyfitting_result.valid) {
				std::cout << "[AutonomousMode] Valid polyfitting result received" << std::endl;
			}

			// MPC control calculation
			ControlCommand command = m_mpcPlanner->runAutonomousStep ();

			// Apply controls with safety
			applyControlsWithSafety (command, control_counter);

		} catch (const std::exception &e) {
			ERROR_STREAM ("AutonomousMode") << "Autonomous control error: " << e.what ();
			if (m_engineController) {
				m_engineController->set_speed (0); // Safety
			}
		}
	}

	INFO_LOG ("AutonomousMode", "Autonomous control loop ended");
}

void AutonomousMode::applyControlsWithSafety (const ControlCommand &control, int control_counter) {
	if (!m_engineController) return;

	// Convert to hardware values with safety limits
	int throttle_pct = static_cast<int> (std::clamp (control.throttle * 100, 0.0, 25.0));
	int steer_angle = static_cast<int> (std::clamp (control.steer * 45.0, -45.0, 45.0));

	// Apply soft start to throttle
	double target_throttle = throttle_pct / 100.0;
	double final_throttle = applySoftStart (target_throttle);
	int final_throttle_pct = static_cast<int> (final_throttle * 100);

	// Store applied controls for state estimation
	m_lastThrottle = final_throttle;
	m_lastSteering = steer_angle * M_PI / 180.0;

	// Enhanced logging for MPC control mode
	if (control_counter % 40 == 0) {
		std::cout << "[MPC CONTROL] Target=" << throttle_pct << "%, Final=" << final_throttle_pct
		          << "%, Steering=" << steer_angle << "°" << std::endl;
	}

	// Apply to hardware
	m_engineController->set_speed (-final_throttle_pct); // Inverted for motor cross-connection
	m_engineController->set_steering (steer_angle);
}

double AutonomousMode::applySoftStart (double target_throttle) {
	return target_throttle;
}
bool AutonomousMode::getCachedEmergencyStop () {
	// Check if emergency stop is required based on cached data
	return m_emergencyStop.load ();
}

void AutonomousMode::emergencyStop () {
	m_emergencyStop = true;
	if (m_engineController) {
		m_engineController->set_speed (0);
		m_engineController->set_steering (0);
	}
	std::cout << "[EMERGENCY] Emergency stop activated!" << std::endl;
}

void AutonomousMode::emergencyMotorStop () {
	emergencyStop (); // Same implementation for now
}

void AutonomousMode::resetEmergencyStop () {
	m_emergencyStop = false;
}
