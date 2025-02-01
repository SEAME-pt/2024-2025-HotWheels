/**
 * @file ControlsManager.hpp
 * @brief Definition of the ControlsManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the ControlsManager class,
 * which is responsible for managing the car's controls.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include <QObject>
#include <QThread>

/**
 * @brief Class that manages the car's controls.
 * @class ControlsManager inherits from QObject
 */
class ControlsManager : public QObject {
	Q_OBJECT

private:
	/** @brief EngineController object. */
	EngineController m_engineController;
	/** @brief JoysticksController object. */
	JoysticksController *m_manualController;
	/** @brief Thread for the manual controller. */
	QThread *m_manualControllerThread;
	/** @brief Current driving mode. */
	DrivingMode m_currentMode;

public:
	explicit ControlsManager(QObject *parent = nullptr);
	~ControlsManager();
	void setMode(DrivingMode mode);

public slots:
	void drivingModeUpdated(DrivingMode newMode);

signals:
	void directionChanged(CarDirection newDirection);
	void steeringChanged(int newAngle);
};

#endif // CONTROLSMANAGER_HPP
