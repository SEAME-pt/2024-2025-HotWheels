#include "DataManager.hpp"
#include <QDebug>

DataManager::DataManager() {}

DataManager::~DataManager() {}

// CAN Data Handling
void DataManager::handleRpmData(int rawRpm)
{
    int processedRpm = rawRpm;
    m_rpm = processedRpm;
    emit canDataProcessed(m_speed, processedRpm, this->m_clusterMetrics);
}

void DataManager::handleSpeedData(float rawSpeed)
{
    float processedSpeed = rawSpeed;
    if (m_clusterMetrics == ClusterMetrics::Miles) {
        processedSpeed *= 0.621371f;
    }
    m_speed = processedSpeed;
    emit canDataProcessed(processedSpeed, m_rpm, this->m_clusterMetrics);
}

// Engine Data Handling
void DataManager::handleDirectionData(CarDirection rawDirection)
{
    if (m_carDirection != rawDirection) {
        m_carDirection = rawDirection;
        emit engineDataProcessed(rawDirection, m_steeringDirection);
    }
}

void DataManager::handleSteeringData(int rawAngle)
{
    if (m_steeringDirection != rawAngle) {
        m_steeringDirection = rawAngle;
        emit engineDataProcessed(m_carDirection, rawAngle);
    }
}

// System Data Handling
void DataManager::handleTimeData(const QString &time)
{
    if (m_time != time) {
        m_time = time;
        emit systemTimeUpdated(time);
    }
}

void DataManager::handleWifiData(const QString &status, const QString &wifiName)
{
    if (m_wifiStatus != status || m_wifiName != wifiName) {
        m_wifiStatus = status;
        m_wifiName = wifiName;
        emit systemWifiUpdated(status, wifiName);
    }
}

void DataManager::handleTemperatureData(const QString &temperature)
{
    if (m_temperature != temperature) {
        m_temperature = temperature;
        emit systemTemperatureUpdated(temperature);
    }
}

// Battery Data Handling
void DataManager::handleBatteryPercentage(float batteryPercentage)
{
    if (!qFuzzyCompare(batteryPercentage, m_batteryPercentage)) {
        m_batteryPercentage = batteryPercentage;
        emit batteryPercentageUpdated(batteryPercentage);
    }
}

// Driving Mode Handling
void DataManager::setDrivingMode(DrivingMode newMode)
{
    if (m_drivingMode != newMode) {
        m_drivingMode = newMode;
        emit drivingModeUpdated(newMode);
    }
}

void DataManager::toggleDrivingMode()
{
    if (m_drivingMode == DrivingMode::Manual) {
        setDrivingMode(DrivingMode::Automatic);
    } else {
        setDrivingMode(DrivingMode::Manual);
    }
}

// Cluster Theme Handling
void DataManager::setClusterTheme(ClusterTheme newTheme)
{
    if (m_clusterTheme != newTheme) {
        m_clusterTheme = newTheme;
        emit clusterThemeUpdated(newTheme);
    }
}

void DataManager::toggleClusterTheme()
{
    if (m_clusterTheme == ClusterTheme::Dark) {
        setClusterTheme(ClusterTheme::Light);
    } else {
        setClusterTheme(ClusterTheme::Dark);
    }
}

// Cluster Metrics Handling
void DataManager::setClusterMetrics(ClusterMetrics newMetrics)
{
    if (m_clusterMetrics != newMetrics) {
        m_clusterMetrics = newMetrics;
        emit clusterMetricsUpdated(newMetrics);
    }
}

void DataManager::toggleClusterMetrics()
{
    if (m_clusterMetrics == ClusterMetrics::Kilometers) {
        setClusterMetrics(ClusterMetrics::Miles);
    } else {
        setClusterMetrics(ClusterMetrics::Kilometers);
    }
}
