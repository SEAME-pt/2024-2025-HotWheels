#ifndef I2CCONTROLLER_HPP
#define I2CCONTROLLER_HPP

#include "II2CController.hpp"
#include <cstdint>

class I2CController : public II2CController
{
private:
    int i2c_fd_;
    int i2c_addr_;

public:
    I2CController(const char *i2c_device, int address);
    ~I2CController() override;

    void writeRegister(uint8_t reg, uint16_t value) override;
    uint16_t readRegister(uint8_t reg) override;
};

#endif // I2CCONTROLLER_HPP
