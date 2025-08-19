#include "AutomaticMode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

AutomaticMode::AutomaticMode (EngineController *engineController, QObject *parent)
    : QObject (parent),
      m_engineController (engineController), 
      m_controlDataHandler (nullptr),
      m_automaticMode (false),
      m_shouldSlowDown (false) {

    // std::cout << "[AutomaticMode] Initialized" << std::endl;
}

AutomaticMode::~AutomaticMode (void) {
	stopAutomaticControl ();
}

void AutomaticMode::startAutomaticControl () {
	if (m_automaticMode == true) return;

    // Initialize control data handler
    if (!m_controlDataHandler) {
        m_controlDataHandler = new ControlDataHandler();
    }
    m_controlDataHandler->initializeSubscribers();
    
	m_automaticControlThread = QThread::create ([this] () { automaticControlLoop (); });
	m_automaticControlThread->start ();

    // std::cout << "[AutomaticMode] Automatic control started" << std::endl;
    m_automaticMode = true;
}

void AutomaticMode::stopAutomaticControl () {
	if (m_automaticMode == false) return;

    m_automaticMode = false;
    
    // Cleanup control data handler
    if (m_controlDataHandler) {
        m_controlDataHandler->cleanupSubscribers();
        delete m_controlDataHandler;
        m_controlDataHandler = nullptr;
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
                float car_speed = m_controlDataHandler->getCarSpeed();
                auto newShouldSlowDown = m_controlDataHandler->getShouldSlowDown();
                if (m_shouldSlowDown && newShouldSlowDown == false && elapsed_slowdown >= SLOW_DOWN_DURATION_S) {
                    m_shouldSlowDown = false;
                } else if (newShouldSlowDown == true) {
                    m_shouldSlowDown = true;
                    last_slowdown_time = current_time;
                }
                
                auto polyfitting_result = m_controlDataHandler->getPolyfittingResult();
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

ControlCommand AutomaticMode::calculateSteering(const CenterlineResult &centerline_result) {
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