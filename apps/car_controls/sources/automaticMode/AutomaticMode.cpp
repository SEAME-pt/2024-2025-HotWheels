#include "AutomaticMode.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

using Clock = std::chrono::steady_clock;

//! ## --- TEST BLOCK CODE ---

// Defina limiares experimentais, ajuste conforme necessário após teste real.
constexpr double ANGLE_OPEN_CURVE = 10.0;  // graus
constexpr double ANGLE_SHARP_CURVE = 20.0; // graus

struct SegmentType {
    const char* name;
    double LOOK_AHEAD_START;
    double LOOK_AHEAD_END;
    
    SegmentType(const char* n, double start, double end) 
        : name(n), LOOK_AHEAD_START(start), LOOK_AHEAD_END(end) {}
};

static const SegmentType RETA("RETA", 0.1, 0.5);
static const SegmentType CURVA_ABERTA("CURVA_ABERTA", 0.2, 0.7);
static const SegmentType CURVA_FECHADA("CURVA_FECHADA", 0.3, 0.9);


//! ## --- END TEST BLOCK CODE ---

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

//! ##  --- Segment Classification Logic ---
SegmentType classifySegmentType(const std::vector<Point2D> &centerline)
{
    if (centerline.size() < 6)
        return RETA; // Não há pontos suficientes para análise

    // Obtenha três pontos ao longo da centerline
    size_t N = centerline.size();
    size_t idx0 = static_cast<size_t>(N * 0.2);
    size_t idx1 = static_cast<size_t>(N * 0.5);
    size_t idx2 = static_cast<size_t>(N * 0.8);

    const auto &p0 = centerline[idx0];
    const auto &p1 = centerline[idx1];
    const auto &p2 = centerline[idx2];

    // Calcula os vetores de direção
    double dx1 = p1.x - p0.x;
    double dy1 = p1.y - p0.y;
    double dx2 = p2.x - p1.x;
    double dy2 = p2.y - p1.y;

    // Calcula ângulo entre os dois vetores em graus
    double dot = dx1 * dx2 + dy1 * dy2;
    double mag1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    double mag2 = std::sqrt(dx2 * dx2 + dy2 * dy2);

    if (mag1 < 1e-3 || mag2 < 1e-3)
        return RETA; // Segmento muito curto para analisar

    double cosTheta = dot / (mag1 * mag2);
    // Protege de erro numérico
    if (cosTheta > 1.0)
        cosTheta = 1.0;
    if (cosTheta < -1.0)
        cosTheta = -1.0;
    double angle_deg = std::acos(cosTheta) * 180.0 / M_PI;

    // Lógica de decisão pelos limiares definidos
    if (angle_deg < ANGLE_OPEN_CURVE)
        return RETA;
    else if (angle_deg < ANGLE_SHARP_CURVE)
        return CURVA_ABERTA;
    else
        return CURVA_FECHADA;
}

//! ## --- END Segment Classification Logic ---

int AutomaticMode::computeDirectionAngle(const std::vector<Point2D>& centerline)
{
    if (centerline.size() < 6) return 0.0;

    auto segmentType = classifySegmentType(centerline);

    std::cout << "[AutomaticMode] Segment type: " << segmentType.name << std::endl;

    size_t N = centerline.size();
    size_t startIdx = static_cast<size_t>(N * segmentType.LOOK_AHEAD_START);
    size_t endIdx = static_cast<size_t>(N * segmentType.LOOK_AHEAD_END);

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