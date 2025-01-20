#ifndef I2CCONTROLLER_HPP
#define I2CCONTROLLER_HPP

#include <cstdint>
#include <stdexcept>

class I2CController
{
private:
    int i2c_fd_;
    int i2c_addr_;

public:
    I2CController(const char *i2c_device, int address);
    virtual ~I2CController();

    void writeRegister(uint8_t reg, uint16_t value);
    uint16_t readRegister(uint8_t reg);
};

#endif // I2CCONTROLLER_HPP
