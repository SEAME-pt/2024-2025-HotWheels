#include "AutomaticMode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

using Clock = std::chrono::steady_clock;


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
                // float car_speed = m_controlDataHandler->getCarSpeed();
                auto newShouldSlowDown = m_controlDataHandler->getShouldSlowDown();
                if (m_shouldSlowDown && newShouldSlowDown == false && elapsed_slowdown >= SLOW_DOWN_DURATION_S) {
                    m_shouldSlowDown = false;
                } else if (newShouldSlowDown == true) {
                    m_shouldSlowDown = true;
                    last_slowdown_time = current_time;
                }
                
                auto polyfitting_result = m_controlDataHandler->getPolyfittingResult();
                if (polyfitting_result.valid) {
                    ControlCommand command = calculateCommands (polyfitting_result);
                    std::cout << "[AutomaticMode] Applying controls: Throttle = "
                              << command.throttle << ", Steering = " << command.steer << std::endl;
                    applyControls (command);
                } else {
                    // applyControls (ControlCommand{0, 0}); // No valid centerline, stop
                    // std::cerr << "[AutomaticMode] No valid centerline data" << std::endl;
                }
            } catch (const std::exception &e) {
                if (m_engineController) {
                    std::cerr << "[AutomaticMode] Exception in control loop: " << e.what() << std::endl;
                    m_engineController->set_speed (0); // Safety
                }
            }
            last_control_time = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

// float AutomaticMode::calculateThrottle(float new_car_speed, float target_speed) {
//     if (new_car_speed < 0 || target_speed < 0) {
//         std::cerr << "[ControlDataHandler] Invalid speed values: new_car_speed = " << new_car_speed << ", target_speed = " << target_speed << std::endl;
//         return 0; // Invalid speeds, return no throttle
//     }
//     float newThrottle = 0.0f;
//     float speed_diff = target_speed - new_car_speed;

//     // if (this->m_carSpeed - new_car_speed > 0.2f || this->m_carSpeed - new_car_speed > 0.2f) {
//     //     // If the car speed is significantly different, wait for the next loop
//     //     newThrottle = this->m_lastThrottle;
//     // } else 
//     // if (speed_diff > 0.1f && new_car_speed == this->m_carSpeed) {
//     //     // If the car is slower than the target speed, increase throttle
//     //     // newThrottle = this->m_lastThrottle + 0.4f;
//     //     newThrottle = std::max(this->m_lastThrottle + 0.2f, 7.0f);
//     // } else if (speed_diff < -0.1f && target_speed > 0.0f) {
//     //     // If the car is faster than the target speed, decrease throttle
//     //     newThrottle = std::max(this->m_lastThrottle - 0.2f, 7.0f);
//     // } else if (speed_diff < -0.1f) {
//     //     // If the car is faster than the target speed, decrease throttle
//     //     newThrottle = this->m_lastThrottle - 0.2f;
//     // } else {
//     //     // If the speed difference is small, maintain current throttle
//     //     newThrottle = this->m_lastThrottle;
//     // }

//     return STRAIGHT_SPEED_THROTTLE;

//     // DEBUG ALL VALUES USED IN THIS METHOD
//     std::cout << "[AutomaticMode] Current Speed: " << this->m_carSpeed
//                 << ", New Speed: " << new_car_speed
//                 << ", Target Speed: " << target_speed
//                 << ", Speed Difference: " << speed_diff
//                 << ", Last Throttle: " << this->m_lastThrottle
//                 << ", New Throttle: " << newThrottle << std::endl;

//     newThrottle = std::clamp(newThrottle, 0.0f, 50.0f);
//     this->m_lastThrottle = newThrottle;
//     this->m_carSpeed = new_car_speed;
//     return newThrottle;
// }

float AutomaticMode::ffLookup(float target_speed_kmh) {
    if (m_ffTable.empty()) return 0.0f;
    if (target_speed_kmh <= m_ffTable.front().first) return m_ffTable.front().second;
    if (target_speed_kmh >= m_ffTable.back().first)  return m_ffTable.back().second;

    for (size_t i = 1; i < m_ffTable.size(); ++i) {
        auto [s0, t0] = m_ffTable[i-1];
        auto [s1, t1] = m_ffTable[i];
        if (target_speed_kmh <= s1) {
            float a = (target_speed_kmh - s0) / (s1 - s0);
            return t0 + a * (t1 - t0);
        }
    }
    return m_ffTable.back().second;
}

float AutomaticMode::calculateThrottle(float new_car_speed_kmh, float target_speed_kmh) {
    // --- 1) time step ---
    double now_s = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
    double dt = (m_prevTime_s > 0.0) ? (now_s - m_prevTime_s) : 0.0;
    if (dt <= 0.0 || dt > 1.0) { // first run or long gap: reinit
        dt = 0.2; // fall back to your COMMAND_DELAY_S
        m_integrator = 0.0f;
        m_prevDeriv  = 0.0f;
    }
    m_prevTime_s = now_s;

    // --- 2) filter speed measurement (simple IIR) ---
    if (m_speedFilt == 0.0f) m_speedFilt = new_car_speed_kmh; // init to first value
    m_speedFilt = m_speedFilt + m_speedAlpha * (new_car_speed_kmh - m_speedFilt);

    // --- 3) compute error ---
    float error = target_speed_kmh - m_speedFilt;

    // --- 4) PI(+D) terms ---
    float P = m_Kp * error;

    // Integrator with anti-windup: only integrate if not saturated or if error would desaturate
    float u_unsat = P + m_integrator + m_prevDeriv + ffLookup(target_speed_kmh);
    bool atHigh = (u_unsat >= m_throttleMax - 1e-3f);
    bool atLow  = (u_unsat <= m_throttleMin + 1e-3f);
    if (!(atHigh && error > 0.0f) && !(atLow && error < 0.0f)) {
        m_integrator += m_Ki * error * static_cast<float>(dt);
    }

    // Derivative on measurement (filtered) — optional; keep Kd=0 at first
    float derivMeas = -(m_speedFilt - (m_speedFilt - 0.0f)); // placeholder; we use filtered state below
    if (m_Kd > 0.0f) {
        // compute raw derivative of error (on measurement): de/dt ≈ -d(speed)/dt
        static float lastSpeed = 0.0f;
        float ds = (m_speedFilt - lastSpeed);
        lastSpeed = m_speedFilt;
        float rawD = -ds / static_cast<float>(std::max(dt, 1e-3));
        // low-pass filter derivative
        float alpha = static_cast<float>(dt) / (static_cast<float>(dt) + m_dFilterTau);
        m_prevDeriv = (1.0f - alpha) * m_prevDeriv + alpha * (m_Kd * rawD);
    } else {
        m_prevDeriv = 0.0f;
    }

    // --- 5) Sum with feed-forward and clamp ---
    float u = P + m_integrator + m_prevDeriv + ffLookup(target_speed_kmh);
    u = std::clamp(u, m_throttleMin, m_throttleMax);

    // --- 6) Rate-limit (slew) to prevent jumps ---
    float maxStep = m_throttleRatePerSec * static_cast<float>(dt);
    float du = std::clamp(u - m_prevThrottle, -maxStep, maxStep);
    float newThrottle = m_prevThrottle + du;

    // --- 7) Bookkeeping ---
    m_prevThrottle = newThrottle;
    this->m_lastThrottle = newThrottle;
    this->m_carSpeed = new_car_speed_kmh;

    // Optional: zero everything when stopping
    if (target_speed_kmh <= 0.01f) {
        m_integrator = 0.0f;
        m_prevDeriv = 0.0f;
        newThrottle = 0.0f;
        m_prevThrottle = 0.0f;
    }

    return newThrottle;
}


void AutomaticMode::applyControls (const ControlCommand &control) {
	if (!m_engineController) return;

	// Apply to hardware
	m_engineController->set_speed (-control.throttle); // Inverted for motor cross-connection
	m_engineController->set_steering (control.steer);
}

ControlCommand AutomaticMode::calculateCommands(const CenterlineResult &centerline_result) {
    ControlCommand controlCommand;
	    
    try {
        auto angle = computeDirectionAngle(centerline_result.blended);

        controlCommand.steer = angle;
		
        float target_speed = 0.0f;
        if (std::abs(angle) > TURN_ANGLE_THRESHOLD) {
            target_speed = TURN_SPEED;
            // std::cout << "[AutomaticMode] Slowing down due to turn"<< std::endl;
        } else if (this->m_shouldSlowDown) {
            target_speed = SLOW_DOWN_SPEED;
            // std::cout << "[AutomaticMode] Slowing down due to detected object" << std::endl;
        } else {
            target_speed = STRAIGHT_SPEED;
            // std::cout << "[AutomaticMode] Driving straight" << std::endl;
        }

        // float new_car_speed = m_controlDataHandler->getCarSpeed();
        // float throttle = calculateThrottle(new_car_speed, target_speed);
        // controlCommand.throttle = throttle;

        if (angle > TURN_ANGLE_THRESHOLD || angle < -TURN_ANGLE_THRESHOLD) {
    		controlCommand.throttle = TURN_SPEED_THROTTLE;
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

    int final_angle = static_cast<int> (angle_deg);

    this->m_anglesBuffer.push_back(final_angle);

    // make average of the last 5 angles
    if (this->m_anglesBuffer.size() >= 5) {
        int sum = 0;
        for (size_t i = this->m_anglesBuffer.size() - 5; i < this->m_anglesBuffer.size(); i++) {
            sum += this->m_anglesBuffer[i];
        }
        final_angle = sum / 5;
    }

    // int angle_diff = std::abs(final_angle - this->m_lastSteeringAngle);
    // if (angle_diff > 100) {
    //     // If the angle difference is too large, reset to center
    //     final_angle = this->m_lastSteeringAngle;
    // }

    this->m_lastSteeringAngle = final_angle;

    return final_angle;
}