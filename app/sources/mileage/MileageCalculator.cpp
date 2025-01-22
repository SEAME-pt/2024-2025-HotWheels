#include "MileageCalculator.hpp"
#include <QDebug>

MileageCalculator::MileageCalculator()
    : timeInterval(0.01) // 10 ms in seconds
{}

void MileageCalculator::addSpeed(float speed)
{
    speedValues.append(speed);
    // qDebug() << "new speed";
}

double MileageCalculator::calculateDistance()
{
    double totalDistance = 0.0;

    for (float speed : speedValues) {
        // Correct unit conversion from km/h to m/s
        double speedInMetersPerSecond = speed * (1000.0 / 3600.0);
        totalDistance += speedInMetersPerSecond * timeInterval;
    }
    // Clear speed values after calculation
    speedValues.clear();

    return totalDistance;
}
