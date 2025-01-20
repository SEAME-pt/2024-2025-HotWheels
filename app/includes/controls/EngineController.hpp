#ifndef ENGINECONTROLLER_HPP
#define ENGINECONTROLLER_HPP

#include <QObject>
#include "enums.hpp"
#include <atomic>

class EngineController : public QObject
{
    Q_OBJECT

private:
    const int MAX_ANGLE = 180;
    const int SERVO_CENTER_PWM = 345;
    const int SERVO_LEFT_PWM = 345 - 140;
    const int SERVO_RIGHT_PWM = 345 + 140;
    const int STEERING_CHANNEL = 0;

    int servo_bus_fd_;
    int motor_bus_fd_;
    int servo_addr_;
    int motor_addr_;

    std::atomic<bool> m_running;
    std::atomic<int> m_current_speed;
    std::atomic<int> m_current_angle;
    CarDirection m_currentDirection = CarDirection::Stop;
    bool m_disabled = false;

    virtual void write_byte_data(int fd, int reg, int value);
    virtual int read_byte_data(int fd, int reg);

    void set_servo_pwm(int channel, int on_value, int off_value);
    void set_motor_pwm(int channel, int value);

    void init_servo();
    void init_motors();

    void disable();
    bool isDisabled() const;

    void setDirection(CarDirection newDirection);

public:
    EngineController(int servo_addr, int motor_addr, QObject *parent = nullptr);
    ~EngineController();

    void start();
    void stop();
    void set_speed(int speed);
    void set_steering(int angle);

    bool get_is_running() const { return m_running; }
    int get_speed() const { return m_current_speed; }
    int get_angle() const { return m_current_angle; }
    int get_servo_bus_fd() const { return servo_bus_fd_; }
    int get_motor_bus_fd() const { return motor_bus_fd_; }
    int get_servo_addr() const { return servo_addr_; }
    int get_motor_addr() const { return motor_addr_; }
    int get_servo_center_pwm() const { return SERVO_CENTER_PWM; }
    int get_servo_left_pwm() const { return SERVO_LEFT_PWM; }
    int get_servo_right_pwm() const { return SERVO_RIGHT_PWM; }
    int get_steering_channel() const { return STEERING_CHANNEL; }
    int get_max_angle() const { return MAX_ANGLE; }
    int get_current_speed() const { return m_current_speed; }
    int get_current_angle() const { return m_current_angle; }
    bool isDisabledPublic() const { return isDisabled(); }

signals:
    void directionUpdated(CarDirection newDirection);
    void steeringUpdated(int newAngle);
};

#endif // ENGINECONTROLLER_HPP
