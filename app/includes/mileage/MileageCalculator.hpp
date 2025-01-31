#ifndef MILEAGECALCULATOR_HPP
#define MILEAGECALCULATOR_HPP

#include <QElapsedTimer>
#include <QList>
#include "IMileageCalculator.hpp"

class MileageCalculator : public IMileageCalculator
{
public:
    MileageCalculator();
    ~MileageCalculator() = default;
    void addSpeed(float speed) override;
    double calculateDistance() override;

private:
    QList<QPair<float, qint64>> m_speedValues;
    QElapsedTimer m_intervalTimer;
};

#endif // MILEAGECALCULATOR_HPP
