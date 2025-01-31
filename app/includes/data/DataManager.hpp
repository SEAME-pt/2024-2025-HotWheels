#ifndef DATAMANAGER_HPP
#define DATAMANAGER_HPP

#include <QObject>
#include <QString>
#include "ClusterSettingsManager.hpp"
#include "SystemDataManager.hpp"
#include "VehicleDataManager.hpp"
#include "enums.hpp"

class DataManager : public QObject
{
    Q_OBJECT

public:
    explicit DataManager(QObject *parent = nullptr);
    ~DataManager();

public slots:
    // Forwarded slots from subclasses
    void handleRpmData(int rawRpm);
    void handleSpeedData(float rawSpeed);
    void handleSteeringData(int rawAngle);
    void handleDirectionData(CarDirection rawDirection);
    void handleTimeData(const QString &currentDate,
                        const QString &currentTime,
                        const QString &currentDay);
    void handleWifiData(const QString &status, const QString &wifiName);
    void handleTemperatureData(const QString &temperature);
    void handleIpAddressData(const QString &ipAddress);
    void handleBatteryPercentage(float batteryPercentage);
    void handleMileageUpdate(double mileage);
    void toggleDrivingMode();
    void toggleClusterTheme();
    void toggleClusterMetrics();

signals:
    // Forwarded signals from subclasses
    void canDataProcessed(float processedSpeed, int processedRpm);
    void engineDataProcessed(CarDirection processedDirection, int processedAngle);
    void systemTimeUpdated(const QString &currentDate,
                           const QString &currentTime,
                           const QString &currentDay);
    void systemWifiUpdated(const QString &status, const QString &wifiName);
    void systemTemperatureUpdated(const QString &temperature);
    void ipAddressUpdated(const QString &ipAddress);
    void batteryPercentageUpdated(float batteryPercentage);
    void mileageUpdated(double mileage);
    void drivingModeUpdated(DrivingMode newMode);
    void clusterThemeUpdated(ClusterTheme newTheme);
    void clusterMetricsUpdated(ClusterMetrics newMetrics);

private:
    VehicleDataManager *m_vehicleDataManager;
    SystemDataManager *m_systemDataManager;
    ClusterSettingsManager *m_clusterSettingsManager;
};

#endif // DATAMANAGER_HPP
