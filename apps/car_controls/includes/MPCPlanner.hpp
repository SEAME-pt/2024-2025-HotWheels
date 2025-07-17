#ifndef MPCPLANNER_HPP
#define MPCPLANNER_HPP

#include "MPCOptimizer.hpp"
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

class MPCPlanner {

	private:
		std::vector<Eigen::Vector2d>
		_prepareReference (const VehicleState &state,
		                   const std::vector<Eigen::Vector2d> &global_waypoints) const;

		MPCOptimizer _optimizer;

		// Direct integration with enhanced inferencer
		struct CachedVisionData {
				std::vector<Point2D> waypoints;
				LaneInfo lane_info{0.0, 0.0};
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedVisionData;
		// State estimation components
		struct StateEstimator {
				VehicleState m_estimatedState;
				std::mutex m_stateMutex;
				std::chrono::steady_clock::time_point m_lastUpdate;
				bool m_initialized = false;
				std::atomic<bool> m_useRealSensors{false};
				std::atomic<double> m_realVelocity{0.0};
				std::atomic<double> m_realYawRate{0.0};
		} m_stateEstimator;

		// Last applied controls for state estimation
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// Adicionar método para mapear comandos para hardware
		ControlCommand _mapCommandsToHardware (double throttle, double steer) const;

	public:
		static constexpr double DATA_TIMEOUT_MS = 200.0;
		MPCPlanner ();
		MPCPlanner &operator= (const MPCPlanner &orign);
		~MPCPlanner (void);

		std::vector<Point2D> extractWaypointsFromLaneInfo (const LaneInfo &lane_info);
		ControlCommand applySmoothSteering (const ControlCommand &control);
		// Enhanced constructor with direct inferencer integration
		VehicleState getCurrentVehicleState ();
		VehicleState getEnhancedVehicleState ();
		void updateVehicleStateEstimation (double applied_throttle, double applied_steering,
		                                   double dt);
		void integrateRealSensorData ();
		LaneInfo generateStraightTrajectory ();
		std::vector<Point2D> getCachedWaypoints ();
		cv::Mat deserializeMask (const std::string &data);

		ControlCommand plan (const VehicleState &current_state,
		                     const std::vector<Point2D> &global_waypoints,
		                     const LaneInfo *lane_info = NULL);

		// Autonomous step method
		ControlCommand runAutonomousStep ();
};

#endif /* !MPCPlanner */