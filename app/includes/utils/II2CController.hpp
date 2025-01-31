#ifndef II2CCONTROLLER_HPP
#define II2CCONTROLLER_HPP

#include <cstdint>

class II2CController
{
public:
    virtual ~II2CController() = default;
    virtual void writeRegister(uint8_t reg, uint16_t value) = 0;
    virtual uint16_t readRegister(uint8_t reg) = 0;
};

#endif // II2CCONTROLLER_HPP
