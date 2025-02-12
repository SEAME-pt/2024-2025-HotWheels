/*!
 * @file SystemInfoProvider.cpp
 * @brief   Implementation of the SystemInfoProvider class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the SystemInfoProvider
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "SystemInfoProvider.hpp"
#include <QDebug>
#include "SystemCommandExecutor.hpp"

	/*!
	 * @brief Constructor for the SystemInfoProvider class.
	 * @details Allocates a SystemCommandExecutor if the one provided is nullptr.
	 * @param executor The executor to use. If nullptr, a new SystemCommandExecutor is
	 *        allocated.
	 */
SystemInfoProvider::SystemInfoProvider(ISystemCommandExecutor *executor)
	: m_executor(executor ? executor : new SystemCommandExecutor())
	, m_ownExecutor(executor == nullptr)
{}


/*! 
 * @brief Destructor for the SystemInfoProvider class.
 * @details Deletes the executor if it was allocated by the SystemInfoProvider. */
SystemInfoProvider::~SystemInfoProvider()
{
	if (m_ownExecutor)
		delete m_executor;
}

	/*!
	 * @brief Gets the current WiFi status.
	 * @param wifiName The name of the WiFi network we are connected to, or an empty
	 *        string if not connected.
	 * @return The current WiFi status as a QString:
	 *         - "Connected" if connected to a network
	 *         - "Disconnected" if not connected to a network
	 *         - "No interface detected" if no wlan interface is detected
	 */
QString SystemInfoProvider::getWifiStatus(QString &wifiName) const
{
	QString output = m_executor->executeCommand("nmcli -t -f DEVICE,STATE,CONNECTION dev");
	QStringList lines = output.split('\n');

	for (const QString &line : lines) {
		if (line.startsWith("wlan")) {
			QStringList parts = line.split(':');
			if (parts.size() >= 3) {
				wifiName = parts[2];
				return (parts[1] == "connected") ? "Connected" : "Disconnected";
			}
		}
	}
	wifiName.clear();
	return "No interface detected";
}


/*!
 * @brief Gets the current temperature in degrees Celsius.
 * @return The current temperature as a QString, e.g. "45.6°C".
 *         If no temperature is available, returns "N/A".
 */
QString SystemInfoProvider::getTemperature() const
{
	QString tempStr = m_executor->readFile("/sys/class/hwmon/hwmon0/temp1_input").trimmed();

	bool ok;
	double tempMillidegrees = tempStr.toDouble(&ok);
	return ok ? QString("%1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1) : "N/A";
}
/*!
 * @brief Gets the current IP address of the WiFi interface.
 * @return The current IP address as a QString.
 *         If no IP address is available, returns "No IP address".
 */

QString SystemInfoProvider::getIpAddress() const
{
	QString output = m_executor->executeCommand(
		"sh -c \"ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+\\.\\d+\\.\\d+\\.\\d+'\"");

	return output.trimmed().isEmpty() ? "No IP address" : output.trimmed();
}
