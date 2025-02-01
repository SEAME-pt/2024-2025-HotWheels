/**
 * @file SystemManager.hpp
 * @brief Definition of the SystemManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the SystemManager class, which
 * is used to manage the system time, status, and battery.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SYSTEMMANAGER_HPP
#define SYSTEMMANAGER_HPP

#include "BatteryController.hpp"
#include <QDateTime>
#include <QFile>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QTextStream>
#include <QTimer>

/**
 * @brief Class that manages the system time, status, and battery.
 * @class SystemManager inherits from QObject
 */
class SystemManager : public QObject {
	Q_OBJECT

public:
	explicit SystemManager(QObject *parent = nullptr);

signals:
	/**
	 * @brief Signal emitted when the system time is updated.
	 * @param currentDate The current date.
	 * @param currentTime The current time.
	 * @param currentDay The current day.
	 */
	void timeUpdated(const QString &currentDate, const QString &currentTime,
									 const QString &currentDay);
	/**
	 * @brief Signal emitted when the wifi status is updated.
	 * @param status The new wifi status.
	 * @param wifiName The name of the wifi network.
	 */
	void wifiStatusUpdated(const QString &status, const QString &wifiName);
	/**
	 * @brief Signal emitted when the temperature is updated.
	 * @param temperature The new temperature.
	 */
	void temperatureUpdated(const QString &temperature);
	/**
	 * @brief Signal emitted when the battery percentage is updated.
	 * @param batteryPercentage The new battery percentage.
	 */
	void batteryPercentageUpdated(float batteryPercentage);
	/**
	 * @brief Signal emitted when the IP address is updated.
	 * @param ipAddress The new IP address.
	 */
	void ipAddressUpdated(const QString &ipAddress);

private slots:
	void updateTime();
	void updateSystemStatus();

private:
	QString fetchWifiStatus(QString &wifiName) const;
	QString fetchTemperature() const;
	QString fetchIpAddress() const;
	/** @brief Timer to update the system time every second. */
	QTimer *m_timeTimer;
	/** @brief Timer to update the system status every 5 seconds. */
	QTimer *m_statusTimer;
	/** @brief BatteryController to fetch battery percentage. */
	BatteryController *m_batteryController;
};

#endif // SYSTEMMANAGER_HPP
