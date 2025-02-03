#ifndef MOCKMILEAGECALCULATOR_HPP
#define MOCKMILEAGECALCULATOR_HPP

#include "IMileageCalculator.hpp"
#include <gmock/gmock.h>

class MockMileageCalculator : public IMileageCalculator
{
public:
    MOCK_METHOD(void, addSpeed, (float speed), (override));
    MOCK_METHOD(double, calculateDistance, (), (override));
};

#endif // MOCKMILEAGECALCULATOR_HPP
