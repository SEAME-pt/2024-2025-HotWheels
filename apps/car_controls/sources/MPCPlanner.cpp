#include "MPCPlanner.hpp"
#include "Debugger.hpp"

#include <algorithm>
#include <chrono>

static double calculateCTE (const std::vector<double> &coeffs, double x, double y) {
	double x_ref = coeffs[0] * y * y + coeffs[1] * y + coeffs[2];
	return x_ref - x;
}

static double calculateEPSI (const std::vector<double> &coeffs, double y, double yaw) {
	double dx_dy = 2 * coeffs[0] * y + coeffs[1];
	double psi_des = std::atan (dx_dy);
	return yaw - psi_des;
}

static std::vector<double> getPolynomialCoeffs (const std::vector<Point2D> &reference) {
	if (reference.size () < 3) return {0.0, 0.0, 0.0};
	Eigen::MatrixXd A (reference.size (), 3);
	Eigen::VectorXd X (reference.size ());
	for (size_t i = 0; i < reference.size (); ++i) {
		double y = reference[i].y;
		A (i, 0) = y * y;
		A (i, 1) = y;
		A (i, 2) = 1.0;
		X (i) = reference[i].x;
	}
	Eigen::VectorXd coeffs = A.colPivHouseholderQr ().solve (X);
	return {coeffs (0), coeffs (1), coeffs (2)};
}

MPCPlanner &MPCPlanner::operator= (const MPCPlanner &origin) {
	if (this != &origin) *this = origin;
	return *this;
}

MPCPlanner::~MPCPlanner (void) {}

MPCPlanner::MPCPlanner () {
	DEBUG_LOG ("MPCPlanner", "Initialized with direct PolyfitterInferencer integration");
}

ControlCommand MPCPlanner::plan (const VehicleState &current_state,
                                 const std::vector<Point2D> &global_waypoints,
                                 const LaneInfo *lane_info) {
	if (global_waypoints.empty ()) {
		throw std::invalid_argument ("Waypoints list cannot be empty");
	}

	// Convert Point2D to Eigen::Vector2d
	std::vector<Eigen::Vector2d> global_waypoints_eigen;
	global_waypoints_eigen.reserve (global_waypoints.size ());
	for (const auto &pt : global_waypoints) {
		global_waypoints_eigen.emplace_back (pt.x, pt.y);
	}

	// Get local reference in Eigen format
	std::vector<Eigen::Vector2d> local_ref_eigen =
	    _prepareReference (current_state, global_waypoints_eigen);

	// Convert local_ref_eigen to std::vector<Point2D>
	std::vector<Point2D> local_ref;
	local_ref.reserve (local_ref_eigen.size ());
	for (const auto &pt : local_ref_eigen) {
		local_ref.emplace_back (pt.x (), pt.y ());
	}

	// Garantir velocidade mínima para estabilidade
	double current_velocity = std::max (0.5, current_state.velocity);

	// Calcular polinômio da referência local usando Polyfitter
	std::vector<double> poly_coeffs = getPolynomialCoeffs (local_ref);

	double cte0 = 0.0, epsi0 = 0.0;
	if (!poly_coeffs.empty ()) {
		cte0 = calculateCTE (poly_coeffs, 0.0, 0.0);
		epsi0 = calculateEPSI (poly_coeffs, 0.0, 0.0); // Use 0.0 for local vehicle yaw
	}

	// Tratar latência
	double latency = 0.1;
	double steer0 = 0.0, throttle0 = 0.0;
	std::vector<double> state_with_latency = _optimizer._predictStateWithLatency (
	    0.0, 0.0, current_state.yaw, current_velocity, throttle0, steer0, latency);
	state_with_latency.push_back (cte0);
	state_with_latency.push_back (epsi0);

	auto [throttle, steer] =
	    _optimizer.solve (state_with_latency[0], state_with_latency[1], state_with_latency[2],
	                      state_with_latency[3], local_ref, lane_info);

	return _mapCommandsToHardware (throttle, steer);
}

ControlCommand MPCPlanner::_mapCommandsToHardware (double throttle, double steer) const {
	// === MPC FULL AUTHORITY MAPPING ===
	// Servo test confirmed full range capability - removing artificial limitations

	double mapped_throttle = throttle;

	// Enhanced throttle response for better performance
	if (throttle > 0) {
		mapped_throttle =
		    std::min (1.0, throttle * 1.3); // Increased sensitivity for better acceleration
	}

	// FULL RANGE steering mapping - no artificial deadzone
	double mapped_steer = steer;

	// Only apply hardware limits (±45° = ±0.785 rad), no artificial restrictions
	mapped_steer = std::max (MPCConfig::steering_limits[0],
	                         std::min (MPCConfig::steering_limits[1], mapped_steer));

	return ControlCommand{mapped_throttle, mapped_steer};
}

std::vector<Eigen::Vector2d>
MPCPlanner::_prepareReference (const VehicleState &state,
                               const std::vector<Eigen::Vector2d> &global_waypoints) const {

	std::vector<Eigen::Vector2d> local_points;
	const double cos_yaw = cos (-state.yaw); // Negativo para transformação inversa
	const double sin_yaw = sin (-state.yaw);

	for (const auto &wp : global_waypoints) {
		// 1. Translação (mover origem para posição do veículo)
		double dx = wp.x () - state.x;
		double dy = wp.y () - state.y;

		// 2. Rotação (transformar para referencial do veículo)
		double local_x = dx * cos_yaw - dy * sin_yaw;
		double local_y = dx * sin_yaw + dy * cos_yaw;

		// 3. Filtrar pontos atrás do veículo (x < 0)
		if (local_x > 0.0) {
			local_points.emplace_back (local_x, local_y);
		}

		// Limita ao horizonte
		if (local_points.size () >= static_cast<size_t> (MPCConfig::horizon)) break;
	}

	// Se não temos pontos suficientes, gerar referência reta
	if (local_points.size () < 3) {
		local_points.clear ();
		for (int i = 1; i <= MPCConfig::horizon; ++i) {
			local_points.emplace_back (i * 2.0, 0.0); // Pontos a cada 2m à frente
		}
	}

	return local_points;
}

ControlCommand MPCPlanner::runAutonomousStep () {
	// 1. Gerar waypoints a partir do LaneInfo
	std::vector<Point2D> waypoints;
	LaneInfo lane_info;
	VehicleState state = getEnhancedVehicleState ();

	std::cout << "Running autonomous step..." << std::endl;

	// 2. Planejar comando ótimo via MPC
	return plan (state, waypoints, &lane_info);
}

/*!
 * @brief Extract waypoints from lane information
 */
std::vector<Point2D> MPCPlanner::extractWaypointsFromLaneInfo (const LaneInfo &lane_info) {
	std::vector<Point2D> waypoints;

	if (lane_info.isValid) {
		// Generate waypoints based on lane boundaries
		double center_x = (lane_info.left_boundary + lane_info.right_boundary) / 2.0;

		// Generate forward waypoints
		for (int i = 1; i <= MPCConfig::horizon; ++i) {
			double distance_ahead = i * 2.0; // 2m spacing
			waypoints.emplace_back (distance_ahead, center_x + lane_info.lateral_offset);
		}
	}

	return waypoints;
}

ControlCommand MPCPlanner::applySmoothSteering (const ControlCommand &control) {
	static double last_steering = 0.0;
	double max_steering_change = 0.05; // rad per step
	double steering_diff = control.steer - last_steering;

	ControlCommand smooth_control = control;
	if (std::abs (steering_diff) > max_steering_change) {
		smooth_control.steer =
		    last_steering + (steering_diff > 0 ? max_steering_change : -max_steering_change);
	}
	last_steering = smooth_control.steer;

	return smooth_control;
}

VehicleState MPCPlanner::getCurrentVehicleState () {
	return getEnhancedVehicleState ();
}

VehicleState MPCPlanner::getEnhancedVehicleState () {
	std::lock_guard<std::mutex> lock (m_stateEstimator.m_stateMutex);
	auto now = std::chrono::steady_clock::now ();
	double dt = std::chrono::duration<double> (now - m_stateEstimator.m_lastUpdate).count ();
	m_stateEstimator.m_lastUpdate = now;

	if (!m_stateEstimator.m_initialized) {
		m_stateEstimator.m_estimatedState = {0.0, 0.0, 0.0, 0.5};
		m_stateEstimator.m_initialized = true;
		return m_stateEstimator.m_estimatedState;
	}

	dt = std::clamp (dt, 0.001, 0.1);
	double applied_throttle = m_lastThrottle.load ();
	double applied_steering = m_lastSteering.load ();

	updateVehicleStateEstimation (applied_throttle, applied_steering, dt);

	if (m_stateEstimator.m_useRealSensors.load ()) {
		integrateRealSensorData ();
	}

	return m_stateEstimator.m_estimatedState;
}

void MPCPlanner::updateVehicleStateEstimation (double applied_throttle, double applied_steering,
                                               double dt) {
	VehicleState &state = m_stateEstimator.m_estimatedState;

	// Vehicle parameters (tuned for Jetracer)
	const double wheelbase = 0.15; // 15cm wheelbase
	const double max_acceleration = 3.0;
	const double max_deceleration = 4.0;
	const double rolling_resistance = 0.1;
	const double air_resistance = 0.05;
	const double max_velocity = 2.0;
	const double steering_response = 0.8;

	// === Velocity dynamics ===
	double target_acceleration = applied_throttle * max_acceleration;
	double resistance_force = rolling_resistance + air_resistance * state.velocity * state.velocity;
	double net_acceleration = target_acceleration - resistance_force;

	if (net_acceleration > 0) {
		net_acceleration = std::min (net_acceleration, max_acceleration);
	} else {
		net_acceleration = std::max (net_acceleration, -max_deceleration);
	}

	state.velocity += net_acceleration * dt;
	state.velocity = std::clamp (state.velocity, 0.0, max_velocity);

	if (state.velocity > 0.1) {
		state.velocity += (((double)rand () / RAND_MAX) - 0.5) * 0.02;
	}

	// === Position integration ===
	double distance = state.velocity * dt;
	state.x += distance * std::cos (state.yaw);
	state.y += distance * std::sin (state.yaw);

	// === Yaw dynamics ===
	if (std::abs (applied_steering) > 0.01 && state.velocity > 0.1) {
		double turning_radius = wheelbase / std::tan (applied_steering * steering_response);
		double angular_velocity = state.velocity / turning_radius;
		angular_velocity = std::clamp (angular_velocity, -2.0, 2.0);
		state.yaw += angular_velocity * dt;
		state.yaw += (((double)rand () / RAND_MAX) - 0.5) * 0.01;

		while (state.yaw > M_PI)
			state.yaw -= 2.0 * M_PI;
		while (state.yaw < -M_PI)
			state.yaw += 2.0 * M_PI;
	}
}

// === Cached Data Access Methods ===

std::vector<Point2D> MPCPlanner::getCachedWaypoints () {
	std::lock_guard<std::mutex> lock (m_cachedVisionData.mutex);
	auto now = std::chrono::steady_clock::now ();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - m_cachedVisionData.timestamp)
	        .count ();

	if (m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedVisionData.waypoints;
	}

	std::vector<Point2D> fallback_waypoints;
	for (int i = 1; i <= 10; ++i) {
		fallback_waypoints.emplace_back (i * 2.0, 0.0);
	}
	return fallback_waypoints;
}

void MPCPlanner::integrateRealSensorData () {
	VehicleState &state = m_stateEstimator.m_estimatedState;

	if (m_stateEstimator.m_useRealSensors.load ()) {
		// Use vision data for position correction
		std::vector<Point2D> current_waypoints = getCachedWaypoints ();
		if (!current_waypoints.empty () && current_waypoints.size () >= 2) {
			Point2D first_wp = current_waypoints[0];
			Point2D second_wp = current_waypoints[1];
			double expected_yaw = std::atan2 (second_wp.y - first_wp.y, second_wp.x - first_wp.x);

			const double vision_weight = 0.1;
			double yaw_correction = expected_yaw - state.yaw;

			while (yaw_correction > M_PI)
				yaw_correction -= 2.0 * M_PI;
			while (yaw_correction < -M_PI)
				yaw_correction += 2.0 * M_PI;

			state.yaw += vision_weight * yaw_correction;
		}

		// Use applied controls for velocity estimation refinement
		double real_throttle_effect = m_stateEstimator.m_realVelocity.load ();
		if (real_throttle_effect > 0.01) {
			const double feedback_weight = 0.2;
			state.velocity =
			    feedback_weight * real_throttle_effect + (1.0 - feedback_weight) * state.velocity;
		}

		// Add realistic measurement noise
		double vision_noise_x = (((double)rand () / RAND_MAX) - 0.5) * 0.05;
		double vision_noise_y = (((double)rand () / RAND_MAX) - 0.5) * 0.05;
		state.x += vision_noise_x;
		state.y += vision_noise_y;

		double control_noise_vel = (((double)rand () / RAND_MAX) - 0.5) * 0.01;
		double control_noise_yaw = (((double)rand () / RAND_MAX) - 0.5) * 0.005;
		state.velocity += control_noise_vel;
		state.velocity = std::max (0.0, state.velocity);
		state.yaw += control_noise_yaw;

		while (state.yaw > M_PI)
			state.yaw -= 2.0 * M_PI;
		while (state.yaw < -M_PI)
			state.yaw += 2.0 * M_PI;
	}
}

/*!
 * @brief Generate straight trajectory fallback
 */
LaneInfo MPCPlanner::generateStraightTrajectory () {
	LaneInfo fallback_lane;
	fallback_lane.left_boundary = -1.0;
	fallback_lane.right_boundary = 1.0;
	fallback_lane.center_line = 0.0;
	fallback_lane.lateral_offset = 0.0;
	fallback_lane.yaw_error = 0.0;
	fallback_lane.isValid = true;
	return fallback_lane;
}

cv::Mat MPCPlanner::deserializeMask (const std::string &data) {
	std::vector<uchar> buffer (data.begin (), data.end ());
	return cv::imdecode (buffer, cv::IMREAD_GRAYSCALE);
}