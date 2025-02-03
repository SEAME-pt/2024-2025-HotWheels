/*!
 * @file JoysticksController.cpp
 * @brief Implementation of the JoysticksController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the JoysticksController
 *          class, which is used to control the vehicle using a joystick.
 * @note This class uses SDL for joystick input handling.
 *
 * @warning Ensure that SDL is properly installed and configured on your system.
 *
 * @see JoysticksController.hpp for the class definition.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "JoysticksController.hpp"
#include <QDebug>
#include <QThread>

/*!
 * @brief Construct a new JoysticksController object.
 * @param steeringCallback The callback function to update the steering angle.
 * @param speedCallback The callback function to update the speed.
 * @param parent The parent QObject.
 * @details This constructor initializes the JoysticksController object with the
 *          specified callback functions.
 */
JoysticksController::JoysticksController(
    std::function<void(int)> steeringCallback,
    std::function<void(int)> speedCallback, QObject *parent)
    : QObject(parent), m_joystick(nullptr),
      m_updateSteering(std::move(steeringCallback)),
      m_updateSpeed(std::move(speedCallback)), m_running(false) {}

/*!
 * @brief Destroy the JoysticksController object.
 * @details This destructor closes the joystick and quits SDL.
 */
JoysticksController::~JoysticksController() {
  if (m_joystick) {
    SDL_JoystickClose(m_joystick);
  }
  SDL_Quit();
}

/*!
 * @brief Initialize the JoysticksController.
 * @return true If the initialization was successful.
 * @return false If the initialization failed.
 * @details This function initializes the JoysticksController by opening the
 *          joystick and initializing SDL.
 */
bool JoysticksController::init() {
  if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
    qDebug() << "Failed to initialize SDL:" << SDL_GetError();
    return false;
  }

  m_joystick = SDL_JoystickOpen(0);
  if (!m_joystick) {
    qDebug() << "Failed to open joystick.";
    SDL_Quit();
    return false;
  }

  return true;
}

/*!
 * @brief Request the JoysticksController to stop.
 * @details This function requests the JoysticksController to stop.
 */
void JoysticksController::requestStop() { m_running = false; }

/*!
 * @brief Process the input from the joystick.
 * @details This function processes the input from the joystick and updates the
 *          steering and speed accordingly.
 */
void JoysticksController::processInput() {
  m_running = true;

  if (!m_joystick) {
    qDebug() << "Joystick not initialized.";
    emit finished();
    return;
  }

  while (m_running && !QThread::currentThread()->isInterruptionRequested()) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_JOYAXISMOTION) {
        if (e.jaxis.axis == 0) {
          m_updateSteering(static_cast<int>(e.jaxis.value / 32767.0 * 180));
        } else if (e.jaxis.axis == 3) {
          m_updateSpeed(static_cast<int>(e.jaxis.value / 32767.0 * 100));
        }
      }
    }
    QThread::msleep(10);
  }

  // qDebug() << "Joystick controller loop finished.";
  emit finished();
}
