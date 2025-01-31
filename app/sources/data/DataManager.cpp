#include "DataManager.hpp"
#include <QDebug>

DataManager::DataManager(QObject *parent)
    : QObject(parent)
{
    // Initialize the three managers
    m_vehicleDataManager = new VehicleDataManager(this);
    m_systemDataManager = new SystemDataManager(this);
    m_clusterSettingsManager = new ClusterSettingsManager(this);

    // Connect signals from VehicleDataManager
    connect(m_vehicleDataManager,
            &VehicleDataManager::canDataProcessed,
            this,
            &DataManager::canDataProcessed);
    connect(m_vehicleDataManager,
            &VehicleDataManager::engineDataProcessed,
            this,
            &DataManager::engineDataProcessed);
    connect(m_vehicleDataManager,
            &VehicleDataManager::mileageUpdated,
            this,
            &DataManager::mileageUpdated);

    // Connect signals from SystemDataManager
    connect(m_systemDataManager,
            &SystemDataManager::systemTimeUpdated,
            this,
            &DataManager::systemTimeUpdated);
    connect(m_systemDataManager,
            &SystemDataManager::systemWifiUpdated,
            this,
            &DataManager::systemWifiUpdated);
    connect(m_systemDataManager,
            &SystemDataManager::systemTemperatureUpdated,
            this,
            &DataManager::systemTemperatureUpdated);
    connect(m_systemDataManager,
            &SystemDataManager::ipAddressUpdated,
            this,
            &DataManager::ipAddressUpdated);
    connect(m_systemDataManager,
            &SystemDataManager::batteryPercentageUpdated,
            this,
            &DataManager::batteryPercentageUpdated);

    // Connect signals from ClusterSettingsManager
    connect(m_clusterSettingsManager,
            &ClusterSettingsManager::drivingModeUpdated,
            this,
            &DataManager::drivingModeUpdated);
    connect(m_clusterSettingsManager,
            &ClusterSettingsManager::clusterThemeUpdated,
            this,
            &DataManager::clusterThemeUpdated);
    connect(m_clusterSettingsManager,
            &ClusterSettingsManager::clusterMetricsUpdated,
            this,
            &DataManager::clusterMetricsUpdated);
}

DataManager::~DataManager()
{
    delete m_vehicleDataManager;
    delete m_systemDataManager;
    delete m_clusterSettingsManager;
}

// Forward slots to VehicleDataManager
void DataManager::handleRpmData(int rawRpm)
{
    m_vehicleDataManager->handleRpmData(rawRpm);
}
void DataManager::handleSpeedData(float rawSpeed)
{
    m_vehicleDataManager->handleSpeedData(rawSpeed);
    qDebug() << "Speed updated";
}
void DataManager::handleSteeringData(int rawAngle)
{
    m_vehicleDataManager->handleSteeringData(rawAngle);
}
void DataManager::handleDirectionData(CarDirection rawDirection)
{
    m_vehicleDataManager->handleDirectionData(rawDirection);
}
void DataManager::handleMileageUpdate(double mileage)
{
    m_vehicleDataManager->handleMileageUpdate(mileage);
}

// Forward slots to SystemDataManager
void DataManager::handleTimeData(const QString &currentDate,
                                 const QString &currentTime,
                                 const QString &currentDay)
{
    m_systemDataManager->handleTimeData(currentDate, currentTime, currentDay);
}

void DataManager::handleWifiData(const QString &status, const QString &wifiName)
{
    m_systemDataManager->handleWifiData(status, wifiName);
}

void DataManager::handleTemperatureData(const QString &temperature)
{
    m_systemDataManager->handleTemperatureData(temperature);
}

void DataManager::handleIpAddressData(const QString &ipAddress)
{
    m_systemDataManager->handleIpAddressData(ipAddress);
}

void DataManager::handleBatteryPercentage(float batteryPercentage)
{
    m_systemDataManager->handleBatteryPercentage(batteryPercentage);
}

// Forward slots to ClusterSettingsManager
void DataManager::toggleDrivingMode()
{
    m_clusterSettingsManager->toggleDrivingMode();
}
void DataManager::toggleClusterTheme()
{
    m_clusterSettingsManager->toggleClusterTheme();
}
void DataManager::toggleClusterMetrics()
{
    m_clusterSettingsManager->toggleClusterMetrics();
}
