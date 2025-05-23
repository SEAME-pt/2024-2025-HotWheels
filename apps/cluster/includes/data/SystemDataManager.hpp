/*!
 * @file SystemDataManager.hpp
 * @brief Definition of the SystemDataManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the SystemDataManager class, which is
 * responsible for managing the data received from the car's systems.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SYSTEMDATAMANAGER_HPP
#define SYSTEMDATAMANAGER_HPP

#include <QObject>
#include <QString>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QProcessEnvironment>
#include <QJsonDocument>
#include <QJsonObject>
#include <QUrl>
#include <QDebug>

/*!
 * @brief Class that manages the data received from the car's systems.
 * @class SystemDataManager inherits from QObject
 */
class SystemDataManager : public QObject
{
	Q_OBJECT

public:
	explicit SystemDataManager(QObject *parent = nullptr);
	~SystemDataManager();

public slots:
	void handleTimeData(const QString &currentMonth,
						const QString &currentTime,
						const QString &currentDay);
	void handleWifiData(const QString &status, const QString &wifiName);
	void handleTemperatureData(const QString &temperature);
	void handleIpAddressData(const QString &ipAddress);
	void handleBatteryPercentage(float batteryPercentage);

signals:
	void systemTimeUpdated(const QString &currentMonth,
						   const QString &currentTime,
						   const QString &currentDay);
	void systemTemperatureUpdated(const QString &temperature);
	void batteryPercentageUpdated(float batteryPercentage);

private:
	QString m_time = "";
	QString m_wifiName = "";
	QString m_wifiStatus = "";
	QString m_temperature = "";
	QString m_ipAddress = "";
	float m_batteryPercentage = -1.0f;

    QNetworkAccessManager *m_manager;
};

#endif // SYSTEMDATAMANAGER_HPP
