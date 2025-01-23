#include "MileageCalculator.hpp"
#include <QDebug>

MileageCalculator::MileageCalculator()
{
    m_intervalTimer.start();
}

void MileageCalculator::addSpeed(float speed)
{
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

double MileageCalculator::calculateDistance()
{
    // qDebug() << "Calculate distances " << m_speedValues.size();
    double totalDistance = 0.0;

    for (QPair<float, qint64> value : m_speedValues) {
        double speedInMetersPerSecond = value.first * (1000.0 / 3600.0);
        double intervalInSeconds = value.second / 1000.0;
        // qDebug() << "Interval: " << value.second << " in seconds: " << intervalInSeconds;
        totalDistance += speedInMetersPerSecond * intervalInSeconds;
    }

    m_speedValues.clear();

    // qDebug() << "Total distance: " << totalDistance;

    return totalDistance;
}
