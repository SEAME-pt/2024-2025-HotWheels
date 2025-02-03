/*!
 * @file I2CController.hpp
 * @brief Definition of the I2CController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the I2CController class, which
 * is used to control I2C devices.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

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
