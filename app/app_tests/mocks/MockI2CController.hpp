#ifndef MOCKI2CCONTROLLER_HPP
#define MOCKI2CCONTROLLER_HPP

#include "II2CController.hpp"
#include <gmock/gmock.h>

class MockI2CController : public II2CController
{
public:
    MOCK_METHOD(void, writeRegister, (uint8_t reg, uint16_t value), (override));
    MOCK_METHOD(uint16_t, readRegister, (uint8_t reg), (override));
};

#endif // MOCKI2CCONTROLLER_HPP
