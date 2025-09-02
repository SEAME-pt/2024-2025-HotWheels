#ifndef AUTOMATIC_MODE_HPP
#define AUTOMATIC_MODE_HPP

#include <QObject>
#include <QThread>

#include "EngineController.hpp"
#include "ControlDataHandler.hpp"
#include "SpeedController.hpp"
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
        // === Controladores ===
        EngineController *m_engineController;
        ControlDataHandler *m_controlDataHandler;
        SpeedController *m_speedController;

        // === Configurações de Velocidade Adaptativa (Otimizadas para Carga Real) ===
        // Velocidades alvo realistas considerando peso e atrito no tapete TNT
        const float STRAIGHT_TARGET_SPEED = 1.2f;      // km/h - retas (reduzido)
        const float TURN_TARGET_SPEED = 0.8f;          // km/h - curvas normais
        const float SHARP_TURN_TARGET_SPEED = 0.6f;    // km/h - curvas fechadas

        // Limiar de ângulo para detecção de curvas
        const int TURN_ANGLE_THRESHOLD = 12;       // graus
        const int SHARP_TURN_ANGLE_THRESHOLD = 45; // graus

        // Scale factor for steering angle
        const double RIGHT_STEERING_SCALE = 2.9;
        const double LEFT_STEERING_SCALE = 3.0;

        // Max steering angle for the car
        const int MAX_STEERING_ANGLE = 180;

        // Delay between control commands to avoid flooding the controller
        const double COMMAND_DELAY_S = 0.05;  // Reduzido para melhor responsividade

        // Segment of the polyfititng blended centerline to use for calculating steering
        const double LOOK_AHEAD_START = 0.3;
        const double LOOK_AHEAD_END = 0.6;

        // Slow down duration
        const double SLOW_DOWN_DURATION_S = 2.0;

        double computeBrakeFromDistance(double distance_m, double speed_mps) const;

        // Driving flags
        bool m_automaticMode;
        bool m_shouldSlowDown;

		QThread *m_automaticControlThread;

        // === Driving Logic Methods ===
        ControlCommand calculateSteeringAndThrottle(const CenterlineResult &centerline_result, float currentSpeed);
        int computeDirectionAngle(const std::vector<Point2D>& centerline);
		void applyControls (const ControlCommand &control);

        // === Thread Loop Method ===
        void automaticControlLoop ();
};

#endif
