#ifndef IMILEAGECALCULATOR_HPP
#define IMILEAGECALCULATOR_HPP

class IMileageCalculator
{
public:
    virtual ~IMileageCalculator() = default;
    virtual void addSpeed(float speed) = 0;
    virtual double calculateDistance() = 0;
};

#endif // IMILEAGECALCULATOR_HPP
