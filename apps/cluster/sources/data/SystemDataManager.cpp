/*!
 * @file SystemDataManager.cpp
 * @brief
 * @version 0.1
 * @date 2025-02-12
 * @details
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "SystemDataManager.hpp"
#include <QtMath>

SystemDataManager::SystemDataManager(QObject *parent)
	: QObject(parent)
    , m_manager(new QNetworkAccessManager(this))
{}

SystemDataManager::~SystemDataManager() {}

/*!
 * @brief Handle Time data.
 * @param currentDate The current date.
 * @param currentTime The current time.
 * @param currentDay The current day.
 * @details This function processes the time data.
 */
void SystemDataManager::handleTimeData(const QString &currentDate,
									   const QString &currentTime,
									   const QString &currentDay)
{
	m_time = currentTime;
	emit systemTimeUpdated(currentDate, currentTime, currentDay);
}

/*!
 * @brief Handle WiFi data.
 * @param status The WiFi status.
 * @param wifiName The WiFi name.
 * @details This function processes the WiFi data.
 */
void SystemDataManager::handleWifiData(const QString &status, const QString &wifiName)
{
	if (m_wifiStatus != status || m_wifiName != wifiName) {
		m_wifiStatus = status;
		m_wifiName = wifiName;
		emit systemWifiUpdated(status, wifiName);

		QJsonObject json;
		json["wifi"] = wifiName;

		QJsonDocument doc(json);
		QByteArray jsonData = doc.toJson();

		QUrl url("https://cluster-app-a7a39eb57433.herokuapp.com/wifi");

		QNetworkRequest request(url);
		request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        m_manager->post(request,jsonData);
	}
}

/*!
 * @brief Handle Temperature data.
 * @param temperature The temperature data.
 * @details This function processes the temperature data.
 */
void SystemDataManager::handleTemperatureData(const QString &temperature)
{
	if (m_temperature != temperature) {
		m_temperature = temperature;
		emit systemTemperatureUpdated(temperature);

		QString temp = temperature;
		temp.remove("°C");

		QJsonObject json;
		json["temperature"] = temp;

		// Convert the JSON object to a QJsonDocument
		QJsonDocument doc(json);
		QByteArray jsonData = doc.toJson();

		QUrl url("https://cluster-app-a7a39eb57433.herokuapp.com/temperature");

		QNetworkRequest request(url);
		request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        m_manager->post(request,jsonData);
	}
}

/*!
 * @brief Handle IP Address data.
 * @param ipAddress The IP address.
 * @details This function processes the IP address data.
 */
void SystemDataManager::handleIpAddressData(const QString &ipAddress)
{
	if (m_ipAddress != ipAddress) {
		m_ipAddress = ipAddress;
		emit ipAddressUpdated(ipAddress);

		QJsonObject json;
		json["ip"] = ipAddress;

		QJsonDocument doc(json);
		QByteArray jsonData = doc.toJson();

		QUrl url("https://cluster-app-a7a39eb57433.herokuapp.com/ip");

		QNetworkRequest request(url);
		request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        m_manager->post(request,jsonData);
	}
}

/*!
 * @brief Handle Battery Percentage data.
 * @param batteryPercentage The battery percentage.
 * @details This function processes the battery percentage data.
 */
void SystemDataManager::handleBatteryPercentage(float batteryPercentage)
{
	if (!qFuzzyCompare(batteryPercentage, m_batteryPercentage)) {
		m_batteryPercentage = batteryPercentage;
		emit batteryPercentageUpdated(batteryPercentage);

		QString temp = QString::number(batteryPercentage);
		temp.remove("%");

		QJsonObject json;
		json["battery"] = temp;

		QJsonDocument doc(json);
		QByteArray jsonData = doc.toJson();

		QUrl url("https://cluster-app-a7a39eb57433.herokuapp.com/battery");

		QNetworkRequest request(url);
		request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

        m_manager->post(request,jsonData);
	}
}
