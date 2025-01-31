/**
 * @file DataManager.cpp
 * @brief Implementation of the DataManager class for handling various data types.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @details This file contains the implementation of the DataManager class, which is responsible for handling
 *          different types of data such as CAN data, system data, battery data, and driving mode.
 * @note See DataManager.hpp for the class definition.
 * @warning Ensure proper data validation before processing.
 * @see DataManager.hpp
 * @copyright Copyright (c) 2025
 */

#include "DataManager.hpp"
#include <QDebug>

DataManager::DataManager() {}

DataManager::~DataManager() {}

// CAN Data Handling

/**
 * @brief Handles RPM data from the CAN bus.
 * @param rawRpm The raw RPM value.
 * @details This function processes the raw RPM value and emits the canDataProcessed signal.
 */
void DataManager::handleRpmData(int rawRpm) {
  int processedRpm = rawRpm;
  m_rpm = processedRpm;
  emit canDataProcessed(m_speed, processedRpm);
}

/**
 * @brief Handles speed data from the CAN bus.
 * @param rawSpeed The raw speed value.
 * @details This function processes the raw speed value, converts it to miles if necessary, and emits the canDataProcessed signal.
 */
void DataManager::handleSpeedData(float rawSpeed) {
  float processedSpeed = rawSpeed;
  if (m_clusterMetrics == ClusterMetrics::Miles) {
    processedSpeed *= 0.621371f;
  }
  m_speed = processedSpeed;
  emit canDataProcessed(processedSpeed, m_rpm);
}

// Mileage Data Handling

/**
 * @brief Updates the mileage data.
 * @param mileage The new mileage value.
 * @details This function updates the mileage value if it has changed and emits the mileageUpdated signal.
 */
void DataManager::handleMileageUpdate(double mileage) {
  if (!qFuzzyCompare(m_mileage, mileage)) {
    m_mileage = mileage;
    // qDebug() << "Mileage updated" << mileage;
    emit mileageUpdated(m_mileage);
  }
}

// Engine Data Handling

/**
 * @brief Handles car direction data.
 * @param rawDirection The raw direction value.
 * @details This function updates the car direction if it has changed and emits the engineDataProcessed signal.
 */
void DataManager::handleDirectionData(CarDirection rawDirection) {
  if (m_carDirection != rawDirection) {
    m_carDirection = rawDirection;
    emit engineDataProcessed(rawDirection, m_steeringDirection);
  }
}

/**
 * @brief Handles steering angle data.
 * @param rawAngle The raw steering angle value.
 * @details This function updates the steering angle if it has changed and emits the engineDataProcessed signal.
 */
void DataManager::handleSteeringData(int rawAngle) {
  if (m_steeringDirection != rawAngle) {
    m_steeringDirection = rawAngle;
    emit engineDataProcessed(m_carDirection, rawAngle);
  }
}

// System Data Handling

/**
 * @brief Handles system time data.
 * @param currentDate The current date.
 * @param currentTime The current time.
 * @param currentDay The current day.
 * @details This function updates the system time and emits the systemTimeUpdated signal.
 */
void DataManager::handleTimeData(const QString &currentDate,
                                 const QString &currentTime,
                                 const QString &currentDay) {
  this->m_time = currentTime;
  emit this->systemTimeUpdated(currentDate, currentTime, currentDay);
}

/**
 * @brief Handles WiFi status data.
 * @param status The WiFi status.
 * @param wifiName The WiFi name.
 * @details This function updates the WiFi status and name if they have changed and emits the systemWifiUpdated signal.
 */
void DataManager::handleWifiData(const QString &status,
                                 const QString &wifiName) {
  if (m_wifiStatus != status || m_wifiName != wifiName) {
    m_wifiStatus = status;
    m_wifiName = wifiName;
    emit systemWifiUpdated(status, wifiName);
  }
}

/**
 * @brief Handles temperature data.
 * @param temperature The temperature value.
 * @details This function updates the temperature value if it has changed and emits the systemTemperatureUpdated signal.
 */
void DataManager::handleTemperatureData(const QString &temperature) {
  if (m_temperature != temperature) {
    m_temperature = temperature;
    emit systemTemperatureUpdated(temperature);
  }
}

/**
 * @brief Handles IP address data.
 * @param ipAddress The IP address.
 * @details This function updates the IP address if it has changed and emits the ipAddressUpdated signal.
 */
void DataManager::handleIpAddressData(const QString &ipAddress) {
  if (m_ipAddress != ipAddress) {
    m_ipAddress = ipAddress;
    emit ipAddressUpdated(ipAddress);
  }
}

// Battery Data Handling

/**
 * @brief Handles battery percentage data.
 * @param batteryPercentage The battery percentage value.
 * @details This function updates the battery percentage if it has changed and emits the batteryPercentageUpdated signal.
 */
void DataManager::handleBatteryPercentage(float batteryPercentage) {
  if (!qFuzzyCompare(batteryPercentage, m_batteryPercentage)) {
    m_batteryPercentage = batteryPercentage;
    emit batteryPercentageUpdated(batteryPercentage);
  }
}

// Driving Mode Handling

/**
 * @brief Sets the driving mode.
 * @param newMode The new driving mode.
 * @details This function updates the driving mode if it has changed and emits the drivingModeUpdated signal.
 */
void DataManager::setDrivingMode(DrivingMode newMode) {
  if (m_drivingMode != newMode) {
    m_drivingMode = newMode;
    emit drivingModeUpdated(newMode);
  }
}

/**
 * @brief Toggles the driving mode between Manual and Automatic.
 * @details This function toggles the driving mode and calls setDrivingMode with the new mode.
 */
void DataManager::toggleDrivingMode() {
  if (m_drivingMode == DrivingMode::Manual) {
    setDrivingMode(DrivingMode::Automatic);
  } else {
    setDrivingMode(DrivingMode::Manual);
  }
}

// Cluster Theme Handling

/**
 * @brief Sets the cluster theme.
 * @param newTheme The new cluster theme.
 * @details This function updates the cluster theme if it has changed and emits the clusterThemeUpdated signal.
 */
void DataManager::setClusterTheme(ClusterTheme newTheme) {
  if (m_clusterTheme != newTheme) {
    m_clusterTheme = newTheme;
    emit clusterThemeUpdated(newTheme);
  }
}

/**
 * @brief Toggles the cluster theme between Dark and Light.
 * @details This function toggles the cluster theme and calls setClusterTheme with the new theme.
 */
void DataManager::toggleClusterTheme() {
  if (m_clusterTheme == ClusterTheme::Dark) {
    setClusterTheme(ClusterTheme::Light);
  } else {
    setClusterTheme(ClusterTheme::Dark);
  }
}

// Cluster Metrics Handling

/**
 * @brief Sets the cluster metrics.
 * @param newMetrics The new cluster metrics.
 * @details This function updates the cluster metrics if they have changed and emits the clusterMetricsUpdated signal.
 */
void DataManager::setClusterMetrics(ClusterMetrics newMetrics) {
  if (m_clusterMetrics != newMetrics) {
    m_clusterMetrics = newMetrics;
    emit clusterMetricsUpdated(newMetrics);
  }
}

/**
 * @brief Toggles the cluster metrics between Kilometers and Miles.
 * @details This function toggles the cluster metrics and calls setClusterMetrics with the new metrics.
 */
void DataManager::toggleClusterMetrics() {
  if (m_clusterMetrics == ClusterMetrics::Kilometers) {
    setClusterMetrics(ClusterMetrics::Miles);
  } else {
    setClusterMetrics(ClusterMetrics::Kilometers);
  }
}
