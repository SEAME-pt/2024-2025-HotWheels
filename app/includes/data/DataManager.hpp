/**
 * @file DataManager.hpp
 * @brief Definition of the DataManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the DataManager class, which is
 * responsible for managing the data received from the car's systems.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef DATAMANAGER_HPP
#define DATAMANAGER_HPP

#include "enums.hpp"
#include <QObject>
#include <QString>

/**
 * @brief Class that manages the data received from the car's systems.
 * @class DataManager inherits from QObject
 */
class DataManager : public QObject {
  Q_OBJECT

public:
  DataManager();
  ~DataManager();

public slots:
  void handleRpmData(int rawRpm);
  void handleSpeedData(float rawSpeed);
  void handleSteeringData(int rawAngle);
  void handleDirectionData(CarDirection rawDirection);
  void handleTimeData(const QString &currentDate, const QString &currentTime,
                      const QString &currentDay);
  void handleWifiData(const QString &status, const QString &wifiName);
  void handleTemperatureData(const QString &temperature);
  void handleIpAddressData(const QString &ipAddress);
  void handleBatteryPercentage(float batteryPercentage);
  void handleMileageUpdate(double mileage);
  void setDrivingMode(DrivingMode newMode);
  void setClusterTheme(ClusterTheme newTheme);
  void setClusterMetrics(ClusterMetrics newMetrics);
  void toggleDrivingMode();
  void toggleClusterTheme();
  void toggleClusterMetrics();

signals:
  /**
   * @brief Signal emitted when the processed speed and RPM are updated.
   * @param processedSpeed The processed speed value.
   * @param processedRpm The processed RPM value.
   */
  void canDataProcessed(float processedSpeed, int processedRpm);
  /**
   * @brief Signal emitted when the processed steering angle is updated.
   * @param processedDirection The processed direction value.
   * @param processedAngle The processed angle value.
   */
  void engineDataProcessed(CarDirection processedDirection, int processedAngle);
  /**
   * @brief Signal emitted when the system time is updated.
   * @param currentDate The current date.
   * @param currentTime The current time.
   * @param currentDay The current day.
   */
  void systemTimeUpdated(const QString &currentDate, const QString &currentTime,
                         const QString &currentDay);
  /**
   * @brief Signal emitted when the WiFi status is updated.
   * @param status The WiFi status.
   * @param wifiName The WiFi name.
   */
  void systemWifiUpdated(const QString &status, const QString &wifiName);
  /**
   * @brief Signal emitted when the system temperature is updated.
   * @param temperature The temperature value.
   */
  void systemTemperatureUpdated(const QString &temperature);
  /**
   * @brief Signal emitted when the IP address is updated.
   * @param ipAddress The IP address.
   */
  void ipAddressUpdated(const QString &ipAddress);
  /**
   * @brief Signal emitted when the battery percentage is updated.
   * @param batteryPercentage The battery percentage value.
   */
  void batteryPercentageUpdated(float batteryPercentage);
  /**
   * @brief Signal emitted when the driving mode is updated.
   * @param newMode The new driving mode.
   */
  void mileageUpdated(double mileage);
  /**
   * @brief Signal emitted when the driving mode is updated.
   * @param newMode The new driving mode.
   */
  void drivingModeUpdated(DrivingMode newMode);
  /**
   * @brief Signal emitted when the cluster theme is updated.
   * @param newTheme The new cluster theme.
   */
  void clusterThemeUpdated(ClusterTheme newTheme);
  /**
   * @brief Signal emitted when the cluster metrics are updated.
   * @param newMetrics The new cluster metrics.
   */
  void clusterMetricsUpdated(ClusterMetrics newMetrics);

private:
  /** @brief Processed speed value. */
  float m_speed = 0.0f;
  /** @brief Processed RPM value. */
  int m_rpm = 0;
  /** @brief Processed direction value. */
  CarDirection m_carDirection = CarDirection::Stop;
  /** @brief Processed steering angle value. */
  int m_steeringDirection = 0;
  /** @brief Processed date value. */
  QString m_time = "";
  /** @brief Processed WiFi name value. */
  QString m_wifiName = "";
  /** @brief Processed WiFi status value. */
  QString m_wifiStatus = "";
  /** @brief Processed temperature value. */
  QString m_temperature = "";
  /** @brief Processed IP address value. */
  QString m_ipAddress = "";
  /** @brief Processed battery percentage value. */
  float m_batteryPercentage = -1.0f;
  /** @brief Processed mileage value. */
  double m_mileage = 0.0;
  /** @brief Processed driving mode value. */
  DrivingMode m_drivingMode = DrivingMode::Manual;
  /** @brief Processed cluster theme value. */
  ClusterTheme m_clusterTheme = ClusterTheme::Dark;
  /** @brief Processed cluster metrics value. */
  ClusterMetrics m_clusterMetrics = ClusterMetrics::Kilometers;
};

#endif // DATAMANAGER_HPP
