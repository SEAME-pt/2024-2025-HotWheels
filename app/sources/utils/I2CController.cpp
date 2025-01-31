#include "I2CController.hpp"
#include <cstdio>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>


I2CController::I2CController(const char *i2c_device, int address)
    : i2c_fd_(-1)
    , i2c_addr_(address)
{
    // Open I2C device
    i2c_fd_ = open(i2c_device, O_RDWR);
    if (i2c_fd_ < 0) {
        perror("Failed to open I2C device");
    }

    // Set I2C address
    if (ioctl(i2c_fd_, I2C_SLAVE, i2c_addr_) < 0) {
        perror("Failed to set I2C address");
    }
}

I2CController::~I2CController()
{
    if (i2c_fd_ >= 0) {
        close(i2c_fd_);
    }
}

void I2CController::writeRegister(uint8_t reg, uint16_t value)
{
    uint8_t buffer[3] = {reg, static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value & 0xFF)};
    if (write(i2c_fd_, buffer, 3) != 3) {
        perror("I2C write failed");
    }
}

uint16_t I2CController::readRegister(uint8_t reg)
{
    if (write(i2c_fd_, &reg, 1) != 1) {
        perror("I2C write failed");
    }
    uint8_t buffer[2];
    if (read(i2c_fd_, buffer, 2) != 2) {
        perror("I2C read failed");
    }
    return (buffer[0] << 8) | buffer[1];
}
