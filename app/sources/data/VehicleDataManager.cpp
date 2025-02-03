#include "VehicleDataManager.hpp"
#include <QtMath>

VehicleDataManager::VehicleDataManager(QObject *parent)
    : QObject(parent)
{}

VehicleDataManager::~VehicleDataManager() {}

// CAN Data Handling
void VehicleDataManager::handleRpmData(int rawRpm)
{
    m_rpm = rawRpm;
    emit canDataProcessed(m_speed, rawRpm);
}

void VehicleDataManager::handleSpeedData(float rawSpeed)
{
    float processedSpeed = rawSpeed;
    if (m_clusterMetrics == ClusterMetrics::Miles) {
        processedSpeed *= 0.621371f;
    }
    m_speed = processedSpeed;
    emit canDataProcessed(processedSpeed, m_rpm);
}

// Mileage Data Handling
void VehicleDataManager::handleMileageUpdate(double mileage)
{
    if (!qFuzzyCompare(m_mileage, mileage)) {
        m_mileage = mileage;
        emit mileageUpdated(m_mileage);
    }
}

// Engine Data Handling
void VehicleDataManager::handleDirectionData(CarDirection rawDirection)
{
    if (m_carDirection != rawDirection) {
        m_carDirection = rawDirection;
        emit engineDataProcessed(rawDirection, m_steeringDirection);
    }
}

void VehicleDataManager::handleSteeringData(int rawAngle)
{
    if (m_steeringDirection != rawAngle) {
        m_steeringDirection = rawAngle;
        emit engineDataProcessed(m_carDirection, rawAngle);
    }
}
