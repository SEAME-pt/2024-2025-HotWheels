/*!
 * @file MileageCalculator.cpp
 * @brief Implementation of the MileageCalculator class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the MileageCalculator
 * class, which is used to calculate the distance traveled by the vehicle.
 * @note This class is used to calculate the distance traveled by the vehicle
 * based on the speed values received.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the interval timer is properly started.
 * @see MileageCalculator.hpp
 * @copyright Copyright (c) 2025
 */

#include "MileageCalculator.hpp"
#include <QDebug>

/*!
 * @brief Construct a new MileageCalculator object.
 * @details This constructor initializes the MileageCalculator object with a
 * started interval timer.
 */
MileageCalculator::MileageCalculator() { m_intervalTimer.start(); }

/*!
 * @brief Add a speed value to the calculator.
 * @param speed The speed value to add.
 * @details This function adds a speed value to the calculator with the current
 * interval time.
 */
void MileageCalculator::addSpeed(float speed) {
	if (m_intervalTimer.isValid()) {
		const qint64 interval = m_intervalTimer.restart();
		QPair<float, qint64> newValue;
		newValue.first = speed;
		newValue.second = interval;
		m_speedValues.append(newValue);

	} else {
		qDebug() << "MileageCalculator Interval Timer was not valid";
	}
}

/*!
 * @brief Calculate the distance traveled by the vehicle.
 * @return double The distance traveled by the vehicle.
 * @details This function calculates the distance traveled by the vehicle based
 * on the speed values received.
 */
double MileageCalculator::calculateDistance() {
	// qDebug() << "Calculate distances " << m_speedValues.size();
	double totalDistance = 0.0;

	for (QPair<float, qint64> value : m_speedValues) {
		double speedInMetersPerSecond = value.first * (1000.0 / 3600.0);
		double intervalInSeconds = value.second / 1000.0;
		// qDebug() << "Interval: " << value.second << " in seconds: " <<
		// intervalInSeconds;
		totalDistance += speedInMetersPerSecond * intervalInSeconds;
	}

	m_speedValues.clear();

	// qDebug() << "Total distance: " << totalDistance;

	return totalDistance;
}
