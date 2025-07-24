#ifndef AUTOMATIC_MODE_HPP
#define AUTOMATIC_MODE_HPP

#include <QObject>
#include <QThread>

#include "EngineController.hpp"
#include "LaneCurveFitter.hpp"
#include "../../ZeroMQ/Subscriber.hpp"
#include "CommonTypes.hpp"

class AutomaticMode : public QObject {
	Q_OBJECT

	public:
		// === Constructors and Destructor ===
		AutomaticMode (EngineController *engineController, QObject *parent = nullptr);
		~AutomaticMode (void);

		// === Core Control Methods ===
		void startAutomaticControl ();
		void stopAutomaticControl ();

	private:
        // Speed when turning and going straight
        const int TURN_SPEED_THROTTLE = 19;
        const int STRAIGHT_SPEED_THROTTLE = 23;
        // Angle from which to consider a turn and apply turn throttle
        const int TURN_ANGLE_THRESHOLD = 150;

        // Scale factor for steering angle
        const double RIGHT_STEERING_SCALE = 2.7;
        const double LEFT_STEERING_SCALE = 3.0;

        // Max steering angle for the car
        const int MAX_STEERING_ANGLE = 180;

        // Delay between control commands to avoid flooding the controller
        const double COMMAND_DELAY_S = 0.2;

        // Segment of the polyfititng blended centerline to use for calculating steering
        const double LOOK_AHEAD_START = 0.3;
        const double LOOK_AHEAD_END = 0.7;

        // Subscriber settings
        const std::string POLYFITTING_TOPIC = "polyfitting_result";
        const std::string POLYFITTING_PORT = "tcp://localhost:5569";

        bool m_automaticMode;

        EngineController *m_engineController;
		LaneCurveFitter *m_curveFitter;

		QThread *m_automaticControlThread;

		Subscriber *m_polyfittingSubscriber;

        // === Driving Logic Methods ===
        ControlCommand calculateSteering(const LaneCurveFitter::CenterlineResult &centerline_result);
        int computeDirectionAngle(const std::vector<Point2D>& centerline);
		void applyControls (const ControlCommand &control);

        // === Subscriber Handling Methods ===
		LaneCurveFitter::CenterlineResult getPolyfittingResult();
		std::vector<LaneCurveFitter::LaneCurve> parseLaneArray(const std::string& json_data);
		std::vector<Point2D> parsePointArray(const std::string& json_data, const std::string& field_name);
		LaneCurveFitter::CenterlineResult extractJsonData(std::string data); 

        // === Thread Loop Method ===
        void automaticControlLoop ();
};

#endif