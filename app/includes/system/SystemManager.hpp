/**
 * @file SystemManager.hpp
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

class SystemManager : public QObject {
  Q_OBJECT

public:
  /**
   * Constructor for the SystemManager class.
   * Initializes timers for updating system time and system status.
   * Sets up communication with the BatteryController.
   *
   * @param parent The parent QObject.
   */
  explicit SystemManager(QObject *parent = nullptr);

signals:
  /**
   * Signal emitted when the system time is updated.
   *
   * @param currentDate The current date in "dd-MM-yy" format.
   * @param currentTime The current time in "HH:mm" format.
   * @param currentDay The current day of the week.
   */
  void timeUpdated(const QString &currentDate, const QString &currentTime,
                   const QString &currentDay);

  /**
   * Signal emitted when the WiFi status is updated.
   *
   * @param status The WiFi status (e.g., "Connected", "Disconnected", "No
   * interface detected").
   * @param wifiName The name of the connected WiFi network, if applicable.
   */
  void wifiStatusUpdated(const QString &status, const QString &wifiName);

  /**
   * Signal emitted when the temperature is updated.
   *
   * @param temperature The current temperature (e.g., "25.0°C").
   */
  void temperatureUpdated(const QString &temperature);

  /**
   * Signal emitted when the battery percentage is updated.
   *
   * @param batteryPercentage The current battery percentage (0.0 to 100.0).
   */
  void batteryPercentageUpdated(float batteryPercentage);

  /**
   * Signal emitted when the IP address is updated.
   *
   * @param ipAddress The current IP address.
   */
  void ipAddressUpdated(const QString &ipAddress);

private slots:
  /**
   * Slot to update the system time every second.
   * Emits the updated time, date, and day.
   */
  void updateTime();

  /**
   * Slot to update system status every 5 seconds.
   * Fetches and emits the WiFi status, temperature, battery percentage, and IP
   * address.
   */
  void updateSystemStatus();

private:
  /**
   * Fetches the current WiFi status and network name.
   *
   * @param wifiName Reference to a QString to store the WiFi network name.
   * @return The status of the WiFi connection (e.g., "Connected",
   * "Disconnected").
   */
  QString fetchWifiStatus(QString &wifiName) const;

  /**
   * Fetches the current temperature from a system file.
   *
   * @return The temperature in degrees Celsius (e.g., "25.0°C").
   */
  QString fetchTemperature() const;

  /**
   * Fetches the current IP address of the system.
   *
   * @return The current IP address (e.g., "192.168.1.2").
   */
  QString fetchIpAddress() const;

  QTimer *m_timeTimer;   // Timer to update the system time every second.
  QTimer *m_statusTimer; // Timer to update the system status every 5 seconds.
  BatteryController
      *m_batteryController; // BatteryController to fetch battery percentage.
};

#endif // SYSTEMMANAGER_HPP
