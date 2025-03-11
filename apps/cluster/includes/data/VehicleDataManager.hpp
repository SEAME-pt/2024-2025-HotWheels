/*!
 * @file VehicleDataManager.hpp
 * @brief Definition of the VehicleDataManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the VehicleDataManager class, which is
 * responsible for managing the data received from the car's systems.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef VEHICLEDATAMANAGER_HPP
#define VEHICLEDATAMANAGER_HPP

#include <QObject>
#include <QString>
#include "enums.hpp"

/*!
 * @brief Class that manages the data received from the car's systems.
 * @class VehicleDataManager inherits from QObject
 */
class VehicleDataManager : public QObject
{
	Q_OBJECT

public:
	explicit VehicleDataManager(QObject *parent = nullptr);
	~VehicleDataManager();

public slots:
	void handleRpmData(int rawRpm);
	void handleSpeedData(float rawSpeed);
	void handleDirectionData(CarDirection rawDirection);
	void handleSteeringData(int rawAngle);
	void handleMileageUpdate(double mileage);

signals:
	void canDataProcessed(float processedSpeed, int processedRpm);
	void engineDataProcessed(CarDirection processedDirection, int processedAngle);
	void mileageUpdated(double mileage);

private:
	float m_speed = 0.0f;
	int m_rpm = 0;
	CarDirection m_carDirection = CarDirection::Stop;
	int m_steeringDirection = 0;
	double m_mileage = 0.0;
    ClusterMetrics m_clusterMetrics = ClusterMetrics::Kilometers;
};

#endif // VEHICLEDATAMANAGER_HPP
