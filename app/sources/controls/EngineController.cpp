/**
 * @file EngineController.cpp
 * @brief Implementation of the EngineController class.
 * @details This file contains the implementation of the EngineController class, which is used to control the engines of the vehicle.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * 
 * @warning Ensure that the servo and motor controllers are properly connected and configured on your system.
 * 
 * @see EngineController.hpp for the class definition.
 * 
 * @note This class is used to control the engines of the vehicle.
 * 
 * @copyright Copyright (c) 2025
 */

#include "EngineController.hpp"
#include "PeripheralController.hpp"
#include <QDebug>
#include <atomic>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

/**
 * @brief Clamp a value between a minimum and a maximum. 
 * 
 * @tparam T The type of the value.
 * @param value The value to be clamped.
 * @param min_val The minimum value.
 * @param max_val The maximum value.
 * @return T The clamped value.
 * @details This function clamps a value between a minimum and a maximum.
 */
template <typename T> T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/**
 * @brief Construct a new EngineController object.
 * @details This constructor initializes the EngineController object.
 */
EngineController::EngineController() {}

/**
 * @brief Construct a new EngineController object.
 * @param servo_addr The address of the servo controller.
 * @param motor_addr The address of the motor controller.
 * @param parent The parent QObject.
 * @details This constructor initializes the EngineController object with the specified servo and motor addresses.
 */
EngineController::EngineController(int servo_addr, int motor_addr,
                                   QObject *parent)
    : QObject(parent), m_running(false), m_current_speed(0),
      m_current_angle(0) {
  pcontrol = new PeripheralController(servo_addr, motor_addr);

  pcontrol->init_servo();
  pcontrol->init_motors();
}

/**
 * @brief Destroy the EngineController object.
 * @details This destructor stops the engine and cleans up resources.
 */
EngineController::~EngineController() {
  stop();
  delete pcontrol;
}

/**
 * @brief Start the engine.
 * @details This function sets the running state to true.
 */
void EngineController::start() { m_running = true; }

/**
 * @brief Stop the engine.
 * @details This function sets the running state to false and stops the motor and steering.
 */
void EngineController::stop() {
  m_running = false;
  set_speed(0);
  set_steering(0);
}

/**
 * @brief Set the direction of the car.
 * @param newDirection The new direction of the car.
 * @details This function updates the car's direction and emits a signal if the direction changes.
 */
void EngineController::setDirection(CarDirection newDirection) {
  if (newDirection != this->m_currentDirection) {
    emit this->directionUpdated(newDirection);
    this->m_currentDirection = newDirection;
  }
}

/**
 * @brief Set the speed of the car.
 * @param speed The new speed of the car.
 * @details This function sets the speed of the car, clamping it between -100 and 100, and updates the motor PWM values accordingly.
 */
void EngineController::set_speed(int speed) {

  speed = clamp(speed, -100, 100);
  int pwm_value = static_cast<int>(std::abs(speed) / 100.0 * 4096);

  if (speed >
      0) { // Forward (but actually backward because joysticks are reversed)
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

/**
 * @brief Set the steering angle of the car.
 * @param angle The new angle of the car.
 * @details This function sets the steering angle of the car, clamping it between -MAX_ANGLE and MAX_ANGLE, and updates the servo PWM values accordingly.
 */
void EngineController::set_steering(int angle) {
  angle = clamp(angle, -MAX_ANGLE, MAX_ANGLE);
  int pwm = 0;
  if (angle < 0) {
    pwm = SERVO_CENTER_PWM +
          static_cast<int>((angle / static_cast<float>(MAX_ANGLE)) *
                           (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
  } else if (angle > 0) {
    pwm = SERVO_CENTER_PWM +
          static_cast<int>((angle / static_cast<float>(MAX_ANGLE)) *
                           (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
  } else {
    pwm = SERVO_CENTER_PWM;
  }

  pcontrol->set_servo_pwm(STEERING_CHANNEL, 0, pwm);
  m_current_angle = angle;
  emit this->steeringUpdated(angle);
}