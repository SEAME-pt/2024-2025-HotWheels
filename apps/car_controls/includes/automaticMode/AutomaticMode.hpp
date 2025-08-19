#ifndef AUTOMATIC_MODE_HPP
#define AUTOMATIC_MODE_HPP

#include <QObject>
#include <QThread>

#include "EngineController.hpp"
#include "ControlDataHandler.hpp"
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
        // Default speed values
        const int TURN_SPEED = 1;
        const int STRAIGHT_SPEED = 2;
        
        // Throttle values for different driving conditions
        const int TURN_SPEED_THROTTLE = 20;
        const int STRAIGHT_SPEED_THROTTLE = 20;
        // Angle from which to consider a turn and apply turn throttle
        const int TURN_ANGLE_THRESHOLD = 150;

        // Scale factor for steering angle
        const double RIGHT_STEERING_SCALE = 2.9;
        const double LEFT_STEERING_SCALE = 3.0;

        // Max steering angle for the car
        const int MAX_STEERING_ANGLE = 180;

        // Delay between control commands to avoid flooding the controller
        const double COMMAND_DELAY_S = 0.1;

        // Segment of the polyfititng blended centerline to use for calculating steering
        const double LOOK_AHEAD_START = 0.3;
        const double LOOK_AHEAD_END = 0.6;

        // Slow down duration
        const double SLOW_DOWN_DURATION_S = 2.0;

        // Driving flags
        bool m_automaticMode;
        bool m_shouldSlowDown;

        EngineController *m_engineController;
        ControlDataHandler *m_controlDataHandler;

		QThread *m_automaticControlThread;

        // === Driving Logic Methods ===
        ControlCommand calculateSteering(const CenterlineResult &centerline_result);
        int computeDirectionAngle(const std::vector<Point2D>& centerline);
		void applyControls (const ControlCommand &control);

        // === Thread Loop Method ===
        void automaticControlLoop ();
};

#endif