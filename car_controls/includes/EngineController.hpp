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
    const int SERVO_CENTER_PWM = 340;
    const int SERVO_LEFT_PWM = 340 - 110;
    const int SERVO_RIGHT_PWM = 340 + 130;
    const int STEERING_CHANNEL = 0;

    std::atomic<bool> m_running;
    std::atomic<int> m_current_speed;
    std::atomic<int> m_current_angle;
    CarDirection m_currentDirection = CarDirection::Stop;

    /**
     * Sets the direction of the car and emits the directionUpdated signal if the direction has changed.
     *
     * @param newDirection The new direction to set.
     */
    void setDirection(CarDirection newDirection);

    IPeripheralController *pcontrol;

public:
    /**
     * Default constructor for the EngineController class.
     */
    EngineController();

    /**
     * Constructor for the EngineController class, initializing the motor and servo controllers.
     *
     * @param servo_addr The address of the servo controller.
     * @param motor_addr The address of the motor controller.
     * @param parent The parent QObject for this instance.
     */
    EngineController(int servo_addr, int motor_addr, QObject *parent = nullptr);

    /**
     * Destructor for the EngineController class.
     *
     * Stops the engine and deletes the peripheral controller.
     */
    ~EngineController();

    /**
     * Starts the engine controller, setting the m_running flag to true.
     */
    void start();

    /**
     * Stops the engine controller, setting the m_running flag to false,
     * and setting speed and steering to zero.
     */
    void stop();

    /**
     * Sets the speed of the car, adjusting the motor PWM values based on the speed.
     *
     * @param speed The speed to set, within the range of -100 to 100.
     */
    void set_speed(int speed);

    /**
     * Sets the steering angle of the car, adjusting the servo PWM based on the angle.
     *
     * @param angle The steering angle to set, within the range of -MAX_ANGLE to MAX_ANGLE.
     */
    void set_steering(int angle);

signals:
    /**
     * Signal emitted when the direction of the car is updated.
     *
     * @param newDirection The new direction of the car.
     */
    void directionUpdated(CarDirection newDirection);

    /**
     * Signal emitted when the steering angle of the car is updated.
     *
     * @param newAngle The new steering angle.
     */
    void steeringUpdated(int newAngle);
};

#endif // ENGINECONTROLLER_HPP
