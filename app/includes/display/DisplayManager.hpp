#ifndef DISPLAYMANAGER_HPP
#define DISPLAYMANAGER_HPP

#include <QObject>
#include <QString>
#include "enums.hpp"
#include "ui_CarManager.h"

class DisplayManager : public QObject
{
    Q_OBJECT

public:
    /**
     * Constructor for the DisplayManager class. Initializes the display manager and sets up connections.
     *
     * @param ui Pointer to the user interface for the car manager.
     * @param parent Optional parent object (default is nullptr).
     */
    explicit DisplayManager(Ui::CarManager *ui, QObject *parent = nullptr);

public slots:
    // CAN
    /**
     * Updates the CAN bus data on the UI with the current speed and RPM.
     *
     * @param speed The current speed of the car.
     * @param rpm The current RPM of the car.
     */
    void updateCanBusData(float speed, int rpm);

    // Engine
    /**
     * Updates the engine data on the UI with the current direction and steering angle.
     *
     * @param direction The current car direction (Drive, Reverse, Stop).
     * @param steeringAngle The current steering angle of the car.
     */
    void updateEngineData(CarDirection direction, int steeringAngle);

    // System
    /**
     * Updates the system time on the UI with the current date, time, and day of the week.
     *
     * @param currentDate The current date.
     * @param currentTime The current time.
     * @param currentDay The current day of the week.
     */
    void updateSystemTime(const QString &currentDate,
                          const QString &currentTime,
                          const QString &currentDay);

    /**
     * Updates the Wi-Fi status on the UI with the current connection status and Wi-Fi name.
     *
     * @param status The current Wi-Fi status (e.g., connected, disconnected).
     * @param wifiName The name of the connected Wi-Fi network.
     */
    void updateWifiStatus(const QString &status, const QString &wifiName);

    /**
     * Updates the temperature on the UI with the current temperature value.
     *
     * @param temperature The current temperature.
     */
    void updateTemperature(const QString &temperature);

    /**
     * Updates the battery percentage on the UI with the current battery level.
     *
     * @param batteryPercentage The current battery percentage.
     */
    void updateBatteryPercentage(float batteryPercentage);

    /**
     * Updates the IP address on the UI with the current IP address.
     *
     * @param ipAddress The current IP address of the system.
     */
    void updateIpAddress(const QString &ipAddress);

    // Mileage
    /**
     * Updates the mileage on the UI with the current mileage.
     *
     * @param mileage The current mileage of the car.
     */
    void updateMileage(double mileage);

    // Settings
    /**
     * Updates the driving mode on the UI (manual or automatic).
     *
     * @param newMode The current driving mode (Manual or Automatic).
     */
    void updateDrivingMode(DrivingMode newMode);

    /**
     * Updates the cluster theme on the UI (Dark or Light).
     *
     * @param newTheme The current cluster theme.
     */
    void updateClusterTheme(ClusterTheme newTheme);

    /**
     * Updates the cluster metrics on the UI (Kilometers or Miles).
     *
     * @param newMetrics The current cluster metrics.
     */
    void updateClusterMetrics(ClusterMetrics newMetrics);

signals:
    // Signals to DataManager
    /**
     * Signal emitted when the driving mode is toggled (Manual/Automatic).
     */
    void drivingModeToggled();

    /**
     * Signal emitted when the cluster theme is toggled (Dark/Light).
     */
    void clusterThemeToggled();

    /**
     * Signal emitted when the cluster metrics are toggled (Kilometers/Miles).
     */
    void clusterMetricsToggled();

private:
    Ui::CarManager *m_ui; /**< Pointer to the user interface for the car manager. */
};

#endif // DISPLAYMANAGER_HPP
