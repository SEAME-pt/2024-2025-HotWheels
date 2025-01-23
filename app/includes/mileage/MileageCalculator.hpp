#ifndef MILEAGECALCULATOR_HPP
#define MILEAGECALCULATOR_HPP

#include <QElapsedTimer>
#include <QList>

class MileageCalculator
{
public:
    MileageCalculator();
    ~MileageCalculator() = default;

    void addSpeed(float speed);
    double calculateDistance();

private:
    QList<QPair<float, qint64>> m_speedValues;
    QElapsedTimer m_intervalTimer;
};

#endif // MILEAGECALCULATOR_HPP
