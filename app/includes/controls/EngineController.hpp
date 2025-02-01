/**
 * @file EngineController.hpp
 * @brief Definition of the EngineController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the EngineController class,
 * which is responsible for controlling the car's engine.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ENGINECONTROLLER_HPP
#define ENGINECONTROLLER_HPP

#include "IPeripheralController.hpp"
#include "enums.hpp"
#include <QObject>
#include <atomic>

/**
 * @brief Class that controls the car's engine.
 * @class EngineController inherits from QObject
 */
class EngineController : public QObject {
	Q_OBJECT

private:
	/** @brief Maximum angle for the steering servo. */
	const int MAX_ANGLE = 180;
	/** @brief Center point for the steering servo. */
	const int SERVO_CENTER_PWM = 345;
	/** @brief PWM value for the leftmost steering position. */
	const int SERVO_LEFT_PWM = 345 - 140;
	/** @brief PWM value for the rightmost steering position. */
	const int SERVO_RIGHT_PWM = 345 + 140;
	/** @brief PWM value for the neutral position. */
	const int STEERING_CHANNEL = 0;

	/** @brief Bool to indicate if the engine is running. */
	std::atomic<bool> m_running;
	/** @brief Current speed of the car. */
	std::atomic<int> m_current_speed;
	/** @brief Current angle of the steering servo. */
	std::atomic<int> m_current_angle;
	/** @brief Current direction of the car. */
	CarDirection m_currentDirection = CarDirection::Stop;

	void setDirection(CarDirection newDirection);

	/** @brief Pointer to the peripheral controller. */
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
	/** @brief Signal emitted when the speed is updated. */
	void directionUpdated(CarDirection newDirection);
	/** @brief Signal emitted when the speed is updated. */
	void steeringUpdated(int newAngle);
};

#endif // ENGINECONTROLLER_HPP
