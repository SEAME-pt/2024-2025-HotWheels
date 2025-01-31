/**
 * @file DataManager.hpp
 * @brief
 * @version 0.1
 * @date 2025-01-31
 * @details
 * @note
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

class DataManager : public QObject {
  Q_OBJECT

public:
  /**
   * Constructor for the DataManager class. Initializes the data manager.
   */
  DataManager();

  /**
   * Destructor for the DataManager class. Cleans up any resources.
   */
  ~DataManager();

public slots:
  // CAN Data
  /**
   * Handles raw RPM data, processes it, and emits a signal with the processed
   * data.
   *
   * @param rawRpm The raw RPM data from the CAN bus.
   */
  void handleRpmData(int rawRpm);

  /**
   * Handles raw speed data, processes it based on cluster metrics, and emits a
   * signal with the processed data.
   *
   * @param rawSpeed The raw speed data from the CAN bus.
   */
  void handleSpeedData(float rawSpeed);

  // Engine Data
  /**
   * Handles the steering angle data, updates the steering direction, and emits
   * the processed data.
   *
   * @param rawAngle The raw steering angle data.
   */
  void handleSteeringData(int rawAngle);

  /**
   * Handles the direction data for the car, updates the direction, and emits
   * the processed data.
   *
   * @param rawDirection The raw direction data (e.g., forward, reverse).
   */
  void handleDirectionData(CarDirection rawDirection);

  // System Data
  /**
   * Handles the time data and emits a signal with the updated date, time, and
   * day.
   *
   * @param currentDate The current date as a string.
   * @param currentTime The current time as a string.
   * @param currentDay The current day of the week as a string.
   */
  void handleTimeData(const QString &currentDate, const QString &currentTime,
                      const QString &currentDay);

  /**
   * Handles the Wi-Fi status and name, and emits a signal with the updated
   * information.
   *
   * @param status The current Wi-Fi status (e.g., connected, disconnected).
   * @param wifiName The name of the current Wi-Fi network.
   */
  void handleWifiData(const QString &status, const QString &wifiName);

  /**
   * Handles the temperature data and emits a signal with the updated
   * temperature.
   *
   * @param temperature The current temperature as a string.
   */
  void handleTemperatureData(const QString &temperature);

  /**
   * Handles the IP address data and emits a signal with the updated IP address.
   *
   * @param ipAddress The current IP address as a string.
   */
  void handleIpAddressData(const QString &ipAddress);

  // Battery Data
  /**
   * Handles the battery percentage data, checks if it has changed, and emits a
   * signal with the updated value.
   *
   * @param batteryPercentage The current battery percentage.
   */
  void handleBatteryPercentage(float batteryPercentage);

  // Mileage Data
  /**
   * Handles the mileage data and emits a signal with the updated mileage.
   *
   * @param mileage The current mileage value.
   */
  void handleMileageUpdate(double mileage);

  // Settings Data
  /**
   * Sets the driving mode to the specified mode and emits a signal with the
   * updated mode.
   *
   * @param newMode The new driving mode (Manual or Automatic).
   */
  void setDrivingMode(DrivingMode newMode);

  /**
   * Sets the cluster theme (Dark or Light) and emits a signal with the updated
   * theme.
   *
   * @param newTheme The new cluster theme.
   */
  void setClusterTheme(ClusterTheme newTheme);

  /**
   * Sets the cluster metrics (Kilometers or Miles) and emits a signal with the
   * updated metrics.
   *
   * @param newMetrics The new cluster metrics.
   */
  void setClusterMetrics(ClusterMetrics newMetrics);

  // Slots to handle toggled settings
  /**
   * Toggles between Manual and Automatic driving modes and emits a signal with
   * the updated mode.
   */
  void toggleDrivingMode();

  /**
   * Toggles between Dark and Light cluster themes and emits a signal with the
   * updated theme.
   */
  void toggleClusterTheme();

  /**
   * Toggles between Kilometers and Miles for the cluster metrics and emits a
   * signal with the updated metrics.
   */
  void toggleClusterMetrics();

signals:
  // CAN Data
  /**
   * Signal emitted when CAN data is processed, with the processed speed and
   * RPM.
   */
  void canDataProcessed(float processedSpeed, int processedRpm);

  /**
   * Signal emitted when engine data is processed, with the processed direction
   * and angle.
   */
  void engineDataProcessed(CarDirection processedDirection, int processedAngle);

  // System Data
  /**
   * Signal emitted when the system time is updated.
   */
  void systemTimeUpdated(const QString &currentDate, const QString &currentTime,
                         const QString &currentDay);

  /**
   * Signal emitted when the Wi-Fi status or name is updated.
   */
  void systemWifiUpdated(const QString &status, const QString &wifiName);

  /**
   * Signal emitted when the system temperature is updated.
   */
  void systemTemperatureUpdated(const QString &temperature);

  /**
   * Signal emitted when the IP address is updated.
   */
  void ipAddressUpdated(const QString &ipAddress);

  // Battery Data
  /**
   * Signal emitted when the battery percentage is updated.
   */
  void batteryPercentageUpdated(float batteryPercentage);

  // Mileage Data
  /**
   * Signal emitted when the mileage is updated.
   */
  void mileageUpdated(double mileage);

  // Settings Data
  /**
   * Signal emitted when the driving mode is updated.
   */
  void drivingModeUpdated(DrivingMode newMode);

  /**
   * Signal emitted when the cluster theme is updated.
   */
  void clusterThemeUpdated(ClusterTheme newTheme);

  /**
   * Signal emitted when the cluster metrics are updated.
   */
  void clusterMetricsUpdated(ClusterMetrics newMetrics);

private:
  // CAN Data
  float m_speed = 0.0f;
  int m_rpm = 0;

  // Engine Data
  CarDirection m_carDirection = CarDirection::Stop;
  int m_steeringDirection = 0;

  // System Data
  QString m_time = "";
  QString m_wifiName = "";
  QString m_wifiStatus = "";
  QString m_temperature = "";
  QString m_ipAddress = "";

  // Battery Data
  float m_batteryPercentage = -1.0f;

  // Mileage Data
  double m_mileage = 0.0;

  // Display Preferences
  DrivingMode m_drivingMode = DrivingMode::Manual;
  ClusterTheme m_clusterTheme = ClusterTheme::Dark;
  ClusterMetrics m_clusterMetrics = ClusterMetrics::Kilometers;
};

#endif // DATAMANAGER_HPP
