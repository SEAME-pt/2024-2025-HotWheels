/*!
 * @file CarManager.hpp
 * @brief Definition of the CarManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the CarManager class, which is
 * responsible for managing the car manager application.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CARMANAGER_H
#define CARMANAGER_H

#include "CanBusManager.hpp"
#include "ControlsManager.hpp"
#include "DataManager.hpp"
#include "DisplayManager.hpp"
#include "MileageManager.hpp"
#include "SystemManager.hpp"
#include "../../ZeroMQ/Publisher.hpp"
#include "../../ZeroMQ/Subscriber.hpp"
#include <QMainWindow>
#include <QThread>

QT_BEGIN_NAMESPACE
/*!
 * @namespace Ui
 * @brief Namespace containing the user interface for the car manager.
 */
namespace Ui {
class CarManager;
}
QT_END_NAMESPACE

/*!
 * @brief Class that manages the car manager application.
 * @class CarManager inherits from QMainWindow
 */
class CarManager : public QMainWindow {
	Q_OBJECT

public:
	CarManager(int argc, char **argv, QWidget *parent = nullptr);
	~CarManager();

private:
	/*! @brief Pointer to the user interface for the car manager. */
	Ui::CarManager *ui;
	/*! @brief Pointer to the DataManager instance. */
	DataManager *m_dataManager;
	/*! @brief Pointer to the CanBusManager instance. */
	CanBusManager *m_canBusManager;
	/*! @brief Pointer to the ControlsManager instance. */
	ControlsManager *m_controlsManager;
	/*! @brief Pointer to the DisplayManager instance. */
	DisplayManager *m_displayManager;
	/*! @brief Pointer to the SystemManager instance. */
	SystemManager *m_systemManager;
	/*! @brief Pointer to the MileageManager instance. */
	MileageManager *m_mileageManager;
	/*! @brief Pointer to the frame subscriber. */
	Subscriber *m_inferenceSubscriber;
	/*! @brief Pointer to the thread for the frame subscriber. */
	QThread *m_inferenceSubscriberThread;
	/*! @brief Flag to indicate if the subscriber is running. */
	bool m_running;

	void initializeComponents();
	void initializeDataManager();
	void initializeCanBusManager();
	void initializeControlsManager();
	void initializeDisplayManager();
	void initializeSystemManager();
	void initializeMileageManager();
};

#endif // CARMANAGER_H
