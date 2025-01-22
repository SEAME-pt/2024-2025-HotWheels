#ifndef DATAMANAGER_HPP
#define DATAMANAGER_HPP

#include <QObject>
#include <QString>
#include "enums.hpp"

class DataManager : public QObject
{
    Q_OBJECT

public:
    DataManager();
    ~DataManager();

public slots:
    // CAN Data
    void handleRpmData(int rawRpm);
    void handleSpeedData(float rawSpeed);

    // Engine Data
    void handleSteeringData(int rawAngle);
    void handleDirectionData(CarDirection rawDirection);

    // System Data
    void handleTimeData(const QString &currentDate,
                        const QString &currentTime,
                        const QString &currentDay);
    void handleWifiData(const QString &status, const QString &wifiName);
    void handleTemperatureData(const QString &temperature);
    void handleIpAddressData(const QString &ipAddress);

    // Battery Data
    void handleBatteryPercentage(float batteryPercentage);

    // Mileage Data
    void handleMileageUpdate(double mileage);

    // Settings Data
    void setDrivingMode(DrivingMode newMode);
    void setClusterTheme(ClusterTheme newTheme);
    void setClusterMetrics(ClusterMetrics newMetrics);

    // Slots to handle toggled settings
    void toggleDrivingMode();
    void toggleClusterTheme();
    void toggleClusterMetrics();

signals:
    // CAN Data
    void canDataProcessed(float processedSpeed, int processedRpm);
    void engineDataProcessed(CarDirection processedDirection, int processedAngle);

    // System Data
    void systemTimeUpdated(const QString &currentDate,
                           const QString &currentTime,
                           const QString &currentDay);
    void systemWifiUpdated(const QString &status, const QString &wifiName);
    void systemTemperatureUpdated(const QString &temperature);
    void ipAddressUpdated(const QString &ipAddress);

    // Battery Data
    void batteryPercentageUpdated(float batteryPercentage);

    // Mileage Data
    void mileageUpdated(double mileage);

    // Settings Data
    void drivingModeUpdated(DrivingMode newMode);
    void clusterThemeUpdated(ClusterTheme newTheme);
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
