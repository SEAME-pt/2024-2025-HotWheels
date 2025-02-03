#include "SystemDataManager.hpp"
#include <QtMath>

SystemDataManager::SystemDataManager(QObject *parent)
    : QObject(parent)
{}

SystemDataManager::~SystemDataManager() {}

// Time Data Handling
void SystemDataManager::handleTimeData(const QString &currentDate,
                                       const QString &currentTime,
                                       const QString &currentDay)
{
    m_time = currentTime;
    emit systemTimeUpdated(currentDate, currentTime, currentDay);
}

// WiFi Data Handling
void SystemDataManager::handleWifiData(const QString &status, const QString &wifiName)
{
    if (m_wifiStatus != status || m_wifiName != wifiName) {
        m_wifiStatus = status;
        m_wifiName = wifiName;
        emit systemWifiUpdated(status, wifiName);
    }
}

// Temperature Data Handling
void SystemDataManager::handleTemperatureData(const QString &temperature)
{
    if (m_temperature != temperature) {
        m_temperature = temperature;
        emit systemTemperatureUpdated(temperature);
    }
}

// IP Address Data Handling
void SystemDataManager::handleIpAddressData(const QString &ipAddress)
{
    if (m_ipAddress != ipAddress) {
        m_ipAddress = ipAddress;
        emit ipAddressUpdated(ipAddress);
    }
}

// Battery Data Handling
void SystemDataManager::handleBatteryPercentage(float batteryPercentage)
{
    if (!qFuzzyCompare(batteryPercentage, m_batteryPercentage)) {
        m_batteryPercentage = batteryPercentage;
        emit batteryPercentageUpdated(batteryPercentage);
    }
}
