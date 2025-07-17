#ifndef AUTONOMOUSMODE_HPP
#define AUTONOMOUSMODE_HPP

#include "EngineController.hpp"
#include "LaneCurveFitter.hpp"
#include "MPCPlanner.hpp"
#include "Publisher.hpp"
#include "Subscriber.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#include <QObject>
#include <QProcess>
#include <QThread>

#ifndef DEFAULT_CONSTANT_SPEED_KMH
#define DEFAULT_CONSTANT_SPEED_KMH 2.0
#endif
#ifndef DEFAULT_CONSTANT_SPEED
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6)
#endif
#ifndef DEFAULT_CONSTANT_THROTTLE
#define DEFAULT_CONSTANT_THROTTLE 0.15
#endif

class AutonomousMode : public QObject {
		Q_OBJECT

	private:
		std::atomic<bool> m_autonomousMode;
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// === Engine Controller Reference ===
		EngineController *m_engineController;

		// === Thread Management ===
		QThread *m_autonomousControlThread;

		// === Frame Subscibers and Publishers ===
		Subscriber *m_inferenceSubscriber;

		// === Object Pointers ===
		MPCPlanner *m_mpcPlanner;
		LaneCurveFitter *m_curveFitter;

		static constexpr double CONTROL_RATE = 20.0;
		static constexpr double DATA_TIMEOUT_MS = 200.0;
		static constexpr double VISION_UPDATE_RATE = 10.0;
		static constexpr double OBSTACLE_UPDATE_RATE = 20.0;
		std::atomic<bool> m_emergencyStop{false};

	public:
		// === Constructors and Destructor ===
		AutonomousMode (EngineController *engineController, QObject *parent = nullptr);
		~AutonomousMode (void);

		// === Core Control Methods ===
		void startAutonomousControl ();
		void stopAutonomousControl ();
		void autonomousControlLoop ();

		// === Configuration Methods ===
		void emergencyMotorStop ();
		void emergencyStop ();
		void resetEmergencyStop ();
		bool isEmergencyStopActive () const {
			return m_emergencyStop.load ();
		}
		double applySoftStart (double target_throttle);

	private:
		void initializeMPCComponents ();
		void applyControlsWithSafety (const ControlCommand &control, int control_counter);
		bool getCachedEmergencyStop ();

		LaneCurveFitter::CenterlineResult getPolyfittingResult();
		std::vector<LaneCurveFitter::LaneCurve> parseLaneArray(const std::string& json_data);
		std::vector<Point2D> parsePointArray(const std::string& json_data, const std::string& field_name);
		LaneCurveFitter::CenterlineResult extractJsonData(std::string data); 

	signals:
		void emergencyStopSignal ();
};

#endif /* !AutonomousMode */