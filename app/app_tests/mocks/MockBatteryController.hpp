#ifndef MOCKBATTERYCONTROLLER_HPP
#define MOCKBATTERYCONTROLLER_HPP

#include "IBatteryController.hpp"
#include <gmock/gmock.h>

class MockBatteryController : public IBatteryController
{
public:
    MOCK_METHOD(float, getBatteryPercentage, (), (override));
};

#endif // MOCKBATTERYCONTROLLER_HPP
