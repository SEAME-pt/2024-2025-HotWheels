#ifndef I2CCONTROLLER_HPP
#define I2CCONTROLLER_HPP

#include <cstdint>
#include <stdexcept>

class I2CController
{
private:
    int i2c_fd_;     // File descriptor for the I2C device
    int i2c_addr_;   // I2C device address

public:
    /**
     * Constructor to initialize the I2C communication with the specified device.
     *
     * @param i2c_device The path to the I2C device (e.g., "/dev/i2c-1").
     * @param address The I2C address of the target device.
     */
    I2CController(const char *i2c_device, int address);

    /**
     * Destructor to close the I2C device file descriptor when the object is destroyed.
     */
    virtual ~I2CController();

    /**
     * Write a 16-bit value to a specific register of the I2C device.
     *
     * @param reg The register address on the I2C device.
     * @param value The 16-bit value to be written to the register.
     */
    void writeRegister(uint8_t reg, uint16_t value);

    /**
     * Read a 16-bit value from a specific register of the I2C device.
     *
     * @param reg The register address on the I2C device.
     * @return The 16-bit value read from the register.
     */
    uint16_t readRegister(uint8_t reg);
};

#endif // I2CCONTROLLER_HPP
