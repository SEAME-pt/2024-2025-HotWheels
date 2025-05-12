/*!
 * @file SystemManager.cpp
 * @brief Implementation of the SystemManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the SystemManager class,
 * which is used to manage the system status.
 * @note This class is used to manage the system status, including the time,
 * WiFi, temperature, battery, and IP address.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the WiFi interface is properly configured and the
 * temperature sensor is connected.
 * @see SystemManager.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "SystemManager.hpp"
#include <QDateTime>
#include <QDebug>
#include "BatteryController.hpp"
#include "SystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"

/*!
 * @brief Constructor for the SystemManager class.
 * @details Allocates a BatteryController, SystemInfoProvider, and
 * SystemCommandExecutor if the ones provided are nullptr.
 * @param batteryController The battery controller to use. If nullptr, a new
 * BatteryController is allocated.
 * @param systemInfoProvider The system info provider to use. If nullptr, a new
 * SystemInfoProvider is allocated.
 * @param systemCommandExecutor The system command executor to use. If nullptr,
 * a new SystemCommandExecutor is allocated.
 * @param parent The parent object of this SystemManager.
 */
SystemManager::SystemManager(IBatteryController *batteryController,
							 ISystemInfoProvider *systemInfoProvider,
							 ISystemCommandExecutor *systemCommandExecutor,
							 QObject *parent)
	: QObject(parent)
	, m_batteryController(batteryController ? batteryController : new BatteryController())
	, m_systemInfoProvider(systemInfoProvider ? systemInfoProvider : new SystemInfoProvider())
	, m_systemCommandExecutor(systemCommandExecutor ? systemCommandExecutor
													: new SystemCommandExecutor())
	, m_ownBatteryController(batteryController == nullptr)
	, m_ownSystemInfoProvider(systemInfoProvider == nullptr)
	, m_ownSystemCommandExecutor(systemCommandExecutor == nullptr)
{}

/*!
 * @brief Destructor for the SystemManager class.
 * @details Calls shutdown() to stop all threads and then deletes the
 * BatteryController, SystemInfoProvider, and SystemCommandExecutor objects if
 * they were allocated by the SystemManager.
 */
SystemManager::~SystemManager()
{
	shutdown();
	if (m_ownBatteryController)
		delete m_batteryController;
	if (m_ownSystemInfoProvider)
		delete m_systemInfoProvider;
	if (m_ownSystemCommandExecutor)
		delete m_systemCommandExecutor;
}

/*!
 * @brief Initializes the SystemManager object.
 * @details This method initializes the SystemManager object by starting two
 * timers: one to update the time every second and another to update the system
 * status every 5 seconds. It also calls updateSystemStatus() to update the
 * system status immediately.
 */
void SystemManager::initialize()
{
	connect(&m_timeTimer, &QTimer::timeout, this, &SystemManager::updateTime);
	connect(&m_statusTimer, &QTimer::timeout, this, &SystemManager::updateSystemStatus);
	m_timeTimer.start(1000);
	m_statusTimer.start(5000);
	updateSystemStatus();
}


/*!
 * @brief Shuts down the SystemManager object.
 * @details This method stops the time and status timers to halt periodic updates.
 */
void SystemManager::shutdown()
{
	m_timeTimer.stop();
	m_statusTimer.stop();
}

/*!
 * @brief Updates the current time.
 * @details This function retrieves the current date and time and emits the
 * timeUpdated signal with the formatted date, time, and weekday.
 */
void SystemManager::updateTime()
{
	QDateTime currentDateTime = QDateTime::currentDateTime();
	emit timeUpdated(currentDateTime.toString("dd-MM-yy"),
					 currentDateTime.toString("HH:mm"),
					 currentDateTime.toString("dddd"));
}

/*!
 * @brief Updates the system status.
 * @details This function updates the system status by calling the getters on the

 * SystemInfoProvider and BatteryController objects and emitting the corresponding
 * signals. It does not block and is intended to be called regularly.
 */
void SystemManager::updateSystemStatus()
{
	QString wifiName;
	emit wifiStatusUpdated(m_systemInfoProvider->getWifiStatus(wifiName), wifiName);
	emit temperatureUpdated(m_systemInfoProvider->getTemperature());
	emit batteryPercentageUpdated(m_batteryController->getBatteryPercentage());
	//emit ipAddressUpdated(m_systemInfoProvider->getIpAddress());
}
