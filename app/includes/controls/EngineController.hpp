#ifndef ENGINECONTROLLER_HPP
#define ENGINECONTROLLER_HPP

#include <QObject>
#include "enums.hpp"
#include <atomic>
#include "IPeripheralController.hpp"

class EngineController : public QObject
{
    Q_OBJECT

private:
    const int MAX_ANGLE = 180;
    const int SERVO_CENTER_PWM = 345;
    const int SERVO_LEFT_PWM = 345 - 140;
    const int SERVO_RIGHT_PWM = 345 + 140;
    const int STEERING_CHANNEL = 0;

    std::atomic<bool> m_running;
    std::atomic<int> m_current_speed;
    std::atomic<int> m_current_angle;
    CarDirection m_currentDirection = CarDirection::Stop;

    void setDirection(CarDirection newDirection);

    IPeripheralController *pcontrol;

public:
    EngineController();
    EngineController(int servo_addr, int motor_addr, QObject *parent = nullptr);
    ~EngineController();

    void start();
    void stop();
    void set_speed(int speed);
    void set_steering(int angle);

signals:
    void directionUpdated(CarDirection newDirection);
    void steeringUpdated(int newAngle);
};

#endif // ENGINECONTROLLER_HPP
