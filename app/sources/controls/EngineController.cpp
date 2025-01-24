#include "EngineController.hpp"
#include <QDebug>
#include <atomic>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "PeripheralController.hpp"

template<typename T>
T clamp(T value, T min_val, T max_val)
{
    return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

EngineController::EngineController() {}

EngineController::EngineController(int servo_addr, int motor_addr, QObject *parent)
    : QObject(parent)
    , m_running(false)
    , m_current_speed(0)
    , m_current_angle(0)
{
    pcontrol = new PeripheralController(servo_addr, motor_addr);

    pcontrol->init_servo();
    pcontrol->init_motors();
}

EngineController::~EngineController()
{
    stop();
    delete pcontrol;
}

void EngineController::start()
{
    m_running = true;
}

void EngineController::stop()
{
    m_running = false;
    set_speed(0);
    set_steering(0);
}


void EngineController::setDirection(CarDirection newDirection)
{
    if (newDirection != this->m_currentDirection) {
        emit this->directionUpdated(newDirection);
        this->m_currentDirection = newDirection;
    }
}

void EngineController::set_speed(int speed)
{

    speed = clamp(speed, -100, 100);
    int pwm_value = static_cast<int>(std::abs(speed) / 100.0 * 4096);

    if (speed > 0) { // Forward (but actually backward because joysticks are reversed)
        pcontrol->set_motor_pwm(0, pwm_value);
        pcontrol->set_motor_pwm(1, 0);
        pcontrol->set_motor_pwm(2, pwm_value);
        pcontrol->set_motor_pwm(5, pwm_value);
        pcontrol->set_motor_pwm(6, 0);
        pcontrol->set_motor_pwm(7, pwm_value);
        setDirection(CarDirection::Reverse);
    } else if (speed < 0) { // Backwards
        pcontrol->set_motor_pwm(0, pwm_value);
        pcontrol->set_motor_pwm(1, pwm_value);
        pcontrol->set_motor_pwm(2, 0);
        pcontrol->set_motor_pwm(5, 0);
        pcontrol->set_motor_pwm(6, pwm_value);
        pcontrol->set_motor_pwm(7, pwm_value);
        setDirection(CarDirection::Drive);
    } else { // Stop
        for (int channel = 0; channel < 9; ++channel)
            pcontrol->set_motor_pwm(channel, 0);
        setDirection(CarDirection::Stop);
    }
    m_current_speed = speed;
}

void EngineController::set_steering(int angle)
{
    angle = clamp(angle, -MAX_ANGLE, MAX_ANGLE);
    int pwm = 0;
    if (angle < 0) {
        pwm = SERVO_CENTER_PWM
              + static_cast<int>((angle / static_cast<float>(MAX_ANGLE))
                                 * (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
    } else if (angle > 0) {
        pwm = SERVO_CENTER_PWM
              + static_cast<int>((angle / static_cast<float>(MAX_ANGLE))
                                 * (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
    } else {
        pwm = SERVO_CENTER_PWM;
    }

    pcontrol->set_servo_pwm(STEERING_CHANNEL, 0, pwm);
    m_current_angle = angle;
    emit this->steeringUpdated(angle);
}
