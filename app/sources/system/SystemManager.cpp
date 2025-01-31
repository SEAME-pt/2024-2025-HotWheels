/**
 * @file SystemManager.cpp
 * @brief Implementation of the SystemManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the SystemManager class, which is used to manage the system status.
 * @note This class is used to manage the system status, including the time, WiFi, temperature, battery, and IP address.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the WiFi interface is properly configured and the temperature sensor is connected.
 * @see SystemManager.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "SystemManager.hpp"

/**
 * @brief Construct a new SystemManager object.
 * @param parent The parent QObject.
 * @details This constructor initializes the SystemManager object with the specified parent.
 */
SystemManager::SystemManager(QObject *parent)
    : QObject(parent), m_timeTimer(new QTimer(this)),
      m_statusTimer(new QTimer(this)),
      m_batteryController(new BatteryController("/dev/i2c-1", 0x41, this)) {
  QTimer::singleShot(0, this, &SystemManager::updateSystemStatus);

  // Update time every second
  connect(m_timeTimer, &QTimer::timeout, this, &SystemManager::updateTime);
  m_timeTimer->start(1000);

  // Update system status (WiFi, temperature, battery) every 5 seconds
  connect(m_statusTimer, &QTimer::timeout, this,
          &SystemManager::updateSystemStatus);
  m_statusTimer->start(5000);
}

/**
 * @brief Destroy the SystemManager object.
 * @details This destructor cleans up the resources used by the SystemManager.
 */
void SystemManager::updateTime() {
  QDateTime currentDateTime = QDateTime::currentDateTime();
  QString currentDate = currentDateTime.toString("dd-MM-yy");
  QString currentTime = currentDateTime.toString("HH:mm");
  QString currentDay = currentDateTime.toString("dddd");

  emit timeUpdated(currentDate, currentTime, currentDay);
}

/**
 * @brief Update the system status.
 * @details This function updates the system status, including the WiFi, temperature, battery, and IP address.
 */
void SystemManager::updateSystemStatus() {
  // Fetch and emit WiFi status
  QString wifiName;
  QString wifiStatus = fetchWifiStatus(wifiName);
  emit wifiStatusUpdated(wifiStatus, wifiName);

  // Fetch and emit temperature
  QString temperature = fetchTemperature();
  emit temperatureUpdated(temperature);

  // Fetch and emit battery percentage
  float batteryPercentage = m_batteryController->getBatteryPercentage();
  emit batteryPercentageUpdated(batteryPercentage);

  // Fetch and emit IP address
  QString ipAddress = fetchIpAddress();
  emit ipAddressUpdated(ipAddress);
}

/**
 * @brief Fetch the WiFi status.
 * @param wifiName The name of the connected WiFi network.
 * @return QString The WiFi status.
 * @details This function fetches the WiFi status and the name of the connected WiFi network.
 */
QString SystemManager::fetchWifiStatus(QString &wifiName) const {
  QProcess process;
  process.start("nmcli", {"-t", "-f", "DEVICE,STATE,CONNECTION", "dev"});
  process.waitForFinished();

  QString output = process.readAllStandardOutput().trimmed();
  QStringList lines = output.split('\n');

  for (const QString &line : lines) {
    if (line.startsWith("wlan")) { // Assuming WiFi interface starts with 'wlan'
      QStringList parts = line.split(':');
      if (parts.size() >= 3) {
        QString state = parts[1];
        wifiName = parts[2]; // Extract connection name

        if (state == "connected") {
          return "Connected";
        } else {
          wifiName.clear();
          return "Disconnected";
        }
      }
    }
  }
  wifiName.clear();
  return "No interface detected";
}

/**
 * @brief Fetch the temperature.
 * @return QString The temperature.
 * @details This function fetches the temperature from the temperature sensor.
 */
QString SystemManager::fetchTemperature() const {
  QString tempFile = "/sys/class/hwmon/hwmon0/temp1_input";
  QFile tempInput(tempFile);
  if (tempInput.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QTextStream in(&tempInput);
    QString tempStr = in.readLine().trimmed();
    tempInput.close();

    bool ok;
    double tempMillidegrees = tempStr.toDouble(&ok);
    if (ok) {
      return QString("%1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1);
    }
  }
  return "N/A";
}

/**
 * @brief Fetch the IP address.
 * @return QString The IP address.
 * @details This function fetches the IP address of the device.
 */
QString SystemManager::fetchIpAddress() const {
  QProcess process;
  process.start(
      "sh",
      {"-c",
       "ip addr show wlan0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1"});
  process.waitForFinished();

  QString output = process.readAllStandardOutput().trimmed();

  if (!output.isEmpty()) {
    return output; // Return the extracted IP address
  }
  return "No IP address"; // Fallback if no IP address is found
}
