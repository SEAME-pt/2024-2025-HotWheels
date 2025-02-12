/*!
 * @file VehicleDataManager.cpp
 * @brief Implementation of the VehicleDataManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the VehicleDataManager class,
 * which is responsible for handling vehicle data.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "VehicleDataManager.hpp"
#include <QtMath>

/*!
 * @brief Construct a new VehicleDataManager::VehicleDataManager object
 * @param parent The parent QObject.
 * @details This constructor initializes the VehicleDataManager object.
 */
VehicleDataManager::VehicleDataManager(QObject *parent)
    : QObject(parent)
{}

/*!
 * @brief Destroy the VehicleDataManager::VehicleDataManager object
 * @details This destructor cleans up the resources used by the VehicleDataManager.
 */
VehicleDataManager::~VehicleDataManager() {}


/*!
 * @brief Handle Rotation Per Minute data.
 * @param rawRpm The raw rotation per minute data.
 * @details This function processes the RPM data by forwarding it to the
 * canDataProcessed signal.
 * @see VehicleDataManager::canDataProcessed
 */
void VehicleDataManager::handleRpmData(int rawRpm)
{
    m_rpm = rawRpm;
    emit canDataProcessed(m_speed, rawRpm);
}

/*!
 * @brief Handle Speed data.
 * @param rawSpeed The raw speed data.
 * @details This function processes the speed data by forwarding it to the
 * canDataProcessed signal. The speed data is converted to the current cluster
 * metrics (either kilometers or miles).
 * @see VehicleDataManager::canDataProcessed
 * @see VehicleDataManager::setClusterMetrics
 */
void VehicleDataManager::handleSpeedData(float rawSpeed)
{
    float processedSpeed = rawSpeed;
    if (m_clusterMetrics == ClusterMetrics::Miles) {
        processedSpeed *= 0.621371f;
    }
    m_speed = processedSpeed;
    emit canDataProcessed(processedSpeed, m_rpm);
}

/*!
 * @brief Handle mileage data.
 * @param mileage The mileage data.
 * @details This function processes the mileage data by updating the internal
 * mileage value and emitting a mileageUpdated signal if the value has changed.
 * @see VehicleDataManager::mileageUpdated
 */
void VehicleDataManager::handleMileageUpdate(double mileage)
{
    if (!qFuzzyCompare(m_mileage, mileage)) {
        m_mileage = mileage;
        emit mileageUpdated(m_mileage);
    }
}

/*!
 * @brief Handle Direction data.
 * @param rawDirection The raw direction data.
 * @details This function processes the direction data by updating the internal
 * direction value and emitting an engineDataProcessed signal if the value has
 * changed.
 * @see VehicleDataManager::engineDataProcessed
 */
void VehicleDataManager::handleDirectionData(CarDirection rawDirection)
{
    if (m_carDirection != rawDirection) {
        m_carDirection = rawDirection;
        emit engineDataProcessed(rawDirection, m_steeringDirection);
    }
}


/*!
 * @brief Handle Steering data.
 * @param rawAngle The raw steering angle data.
 * @details This function processes the steering data by updating the internal
 * steering direction value and emitting an engineDataProcessed signal if the
 * value has changed.
 * @see VehicleDataManager::engineDataProcessed
 */
void VehicleDataManager::handleSteeringData(int rawAngle)
{
    if (m_steeringDirection != rawAngle) {
        m_steeringDirection = rawAngle;
        emit engineDataProcessed(m_carDirection, rawAngle);
    }
}
