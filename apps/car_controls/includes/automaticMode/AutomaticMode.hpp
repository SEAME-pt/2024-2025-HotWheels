#ifndef AUTOMATIC_MODE_HPP
#define AUTOMATIC_MODE_HPP

#include <QObject>
#include <QThread>

#include "EngineController.hpp"
#include "ControlDataHandler.hpp"
#include "CommonTypes.hpp"

class AutomaticMode : public QObject
{
    Q_OBJECT

public:
    // === Constructors and Destructor ===
    AutomaticMode(EngineController *engineController, QObject *parent = nullptr);
    ~AutomaticMode(void);

    // === Core Control Methods ===
    void startAutomaticControl();
    void stopAutomaticControl();

private:
    // Default speed values
    const float TURN_SPEED = 0.3f;
    const float SLOW_DOWN_SPEED = 0.3f;
    const float STRAIGHT_SPEED = 0.5f;

    // Throttle values for different driving conditions
    const int TURN_SPEED_THROTTLE = 20;
    const int STRAIGHT_SPEED_THROTTLE = 22;
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

    // Slow down duration
    const double SLOW_DOWN_DURATION_S = 1.5;

    // --- Controller configuration (tune these) ---
    float m_Kp = 4.0f;         // proportional gain
    float m_Ki = 1.5f;         // integral gain [per (km/h*s)]
    float m_Kd = 0.0f;         // start at 0; add only if needed
    float m_dFilterTau = 0.3f; // derivative low-pass time constant [s]
    float m_speedAlpha = 0.3f; // IIR filter for speed (0..1)

    // --- Limits ---
    float m_throttleMin = 0.0f;
    float m_throttleMax = 50.0f;
    float m_throttleRatePerSec = 80.0f; // max change per second in throttle units

    // --- Feed-forward (optional) ---
    std::vector<std::pair<float, float>> m_ffTable{// {speed_kmh, throttle}
                                                   {0.0f, 0.0f},
                                                   {0.5f, 21.0f},
                                                   {1.0f, 22.0f},
                                                   {1.5f, 23.0f},
                                                   {2.0f, 24.0f}};

    // --- Controller state ---
    float m_integrator = 0.0f;
    float m_prevError = 0.0f;
    float m_prevDeriv = 0.0f;    // filtered derivative state
    float m_prevThrottle = 0.0f; // for rate limiting
    float m_speedFilt = 0.0f;    // filtered speed
    double m_prevTime_s = 0.0;   // last update timestamp

    // Driving flags
    bool m_automaticMode;
    bool m_shouldSlowDown;

    // Driving state
    float m_lastThrottle;
    float m_carSpeed;
    int m_lastSteeringAngle;
    std::vector<int> m_anglesBuffer;

    EngineController *m_engineController;
    ControlDataHandler *m_controlDataHandler;

    QThread *m_automaticControlThread;

    // === Driving Logic Methods ===
    ControlCommand calculateCommands(const CenterlineResult &centerline_result);
    int computeDirectionAngle(const std::vector<Point2D> &centerline);
    void applyControls(const ControlCommand &control);
    float calculateThrottle(float car_speed, float target_speed);
    float ffLookup(float target_speed_kmh);

    // === Thread Loop Method ===
    void automaticControlLoop();
};

#endif