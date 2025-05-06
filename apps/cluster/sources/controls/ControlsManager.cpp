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
ControlsManager::ControlsManager(int argc, char **argv, QObject *parent)
		: QObject(parent), m_serverObject(nullptr),
		m_serverThread(nullptr) {

	// **Client Middleware Interface Thread**
/* 	m_serverThread = QThread::create([this, argc, argv]() {
			m_serverObject = new Publisher(5555);
	});
	m_serverThread->start(); */
	m_serverObject = new Publisher(5555);
}

/*!
 * @brief Destroy the ControlsManager object.
 * @details This destructor stops the joystick controller and waits for the
 * thread to finish.
 */
ControlsManager::~ControlsManager()
{
	if (m_serverThread) {
		m_serverThread->quit();
		m_serverThread->wait();
		delete m_serverThread;
	}
	delete m_serverObject;
}

/*!
 * @brief Update the driving mode of the vehicle.
 * @param newMode The new driving mode of the vehicle.
 * @details This slot is called when the driving mode of the vehicle is changed.
 *          It updates the current driving mode by calling the setMode() method.
 */
void ControlsManager::drivingModeUpdated(DrivingMode newMode) {
	qDebug() << "[ControlsManager] drivingModeUpdated called with: " << (newMode == DrivingMode::Automatic ? "Automatic" : "Manual");

	if (newMode == DrivingMode::Automatic) {
		m_serverObject->setJoystickStatus(false);
	}
	else {
		m_serverObject->setJoystickStatus(true);
	}
}
