/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @details This file contains the implementation of the ControlsManager class,
 *          which is used to manage the controls of the vehicle.
 * @version 0.1
 * @date 2025-01-31
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @note This class is used to manage the controls of the vehicle.
 *
 * @warning Ensure that the EngineController and JoysticksController classes are
 * properly implemented.
 *
 * @see ControlsManager.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <QDebug>

/*!
 * @brief Construct a new ControlsManager object.
 * @param parent The parent QObject.
 * @details This constructor initializes the ControlsManager object.
 */
ControlsManager::ControlsManager(QObject *parent)
    : QObject(parent), m_engineController(0x40, 0x60, this),
      m_manualController(nullptr), m_manualControllerThread(nullptr),
      m_currentMode(DrivingMode::Manual) {
  // Connect EngineController signals to ControlsManager signals
  connect(&m_engineController, &EngineController::directionUpdated, this,
          &ControlsManager::directionChanged);
  connect(&m_engineController, &EngineController::steeringUpdated, this,
          &ControlsManager::steeringChanged);

  // Initialize the joystick controller with callbacks
  m_manualController = new JoysticksController(
      [this](int steering) {
        if (m_currentMode == DrivingMode::Manual) {
          m_engineController.set_steering(steering);
        }
      },
      [this](int speed) {
        if (m_currentMode == DrivingMode::Manual) {
          m_engineController.set_speed(speed);
        }
      });

  if (!m_manualController->init()) {
    qDebug() << "Failed to initialize joystick controller.";
    return;
  }

  // Start the joystick controller in its own thread
  m_manualControllerThread = new QThread(this);
  m_manualController->moveToThread(m_manualControllerThread);

  connect(m_manualControllerThread, &QThread::started, m_manualController,
          &JoysticksController::processInput);
  connect(m_manualController, &JoysticksController::finished,
          m_manualControllerThread, &QThread::quit);

  m_manualControllerThread->start();
}

/*!
 * @brief Destroy the ControlsManager object.
 * @details This destructor stops the joystick controller and waits for the
 * thread to finish.
 */
ControlsManager::~ControlsManager() {
  if (m_manualControllerThread) {
    m_manualController->requestStop();
    m_manualControllerThread->quit();
    m_manualControllerThread->wait();
  }

  delete m_manualController;
}

/*!
 * @brief Set the driving mode.
 * @param mode The new driving mode.
 * @details This slot is called when the driving mode is changed. It updates the
 *          current driving mode and stops the joystick controller if the new
 * mode is Automatic.
 */
void ControlsManager::setMode(DrivingMode mode) {
  if (m_currentMode == mode)
    return;

  m_currentMode = mode;

  // if (m_currentMode == DrivingMode::Automatic) {
  //     qDebug() << "Switched to Automatic mode (joystick disabled).";
  // } else if (m_currentMode == DrivingMode::Manual) {
  //     qDebug() << "Switched to Manual mode (joystick enabled).";
  // }
}

/*!
 * @brief Update the driving mode of the vehicle.
 * @param newMode The new driving mode of the vehicle.
 * @details This slot is called when the driving mode of the vehicle is changed.
 *          It updates the current driving mode by calling the setMode() method.
 */
void ControlsManager::drivingModeUpdated(DrivingMode newMode) {
  setMode(newMode);
}
