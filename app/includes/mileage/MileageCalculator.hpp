#ifndef MILEAGECALCULATOR_HPP
#define MILEAGECALCULATOR_HPP

#include <QList>

class MileageCalculator
{
public:
    MileageCalculator();
    ~MileageCalculator() = default;

    void addSpeed(float speed);
    double calculateDistance();

private:
    QList<float> speedValues;  // Stores speed values in km/h
    const double timeInterval; // Time interval in seconds (10 ms)
};

#endif // MILEAGECALCULATOR_HPP
