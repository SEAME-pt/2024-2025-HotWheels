/**
 * @file I2CController.cpp
 * @brief Implementation of the I2CController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the I2CController class,
 * which is used to control I2C devices.
 * @note This class is used to control I2C devices using the Linux I2C
 * interface.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the I2C device is properly connected and configured.
 * @see I2CController.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "I2CController.hpp"
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

/**
 * @brief Construct a new I2CController object.
 * @param i2c_device The I2C device to use for communication.
 * @param address The I2C address of the device.
 * @throws std::runtime_error if the I2C device cannot be opened or the address
 * cannot be set.
 * @details This constructor initializes the I2CController object with the
 * specified I2C device and address.
 */
I2CController::I2CController(const char *i2c_device, int address)
	: i2c_fd_(-1), i2c_addr_(address) {
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

/**
 * @brief Destroy the I2CController object
 * @details This destructor closes the I2C device.
 */
I2CController::~I2CController() {
  if (i2c_fd_ >= 0) {
	close(i2c_fd_);
  }
}

/**
 * @brief Write a 16-bit value to a register.
 * @param reg The register address to write to.
 * @param value The value to write.
 * @details This function writes a 16-bit value to a register on the I2C device.
 */
void I2CController::writeRegister(uint8_t reg, uint16_t value) {
  std::array<uint8_t, 3> buffer = {reg, static_cast<uint8_t>(value >> 8),
								   static_cast<uint8_t>(value & 0xFF)};
  if (write(i2c_fd_, buffer.data(), buffer.size()) != 3) {
	perror("I2C write failed");
  }
}

/**
 * @brief Read a 16-bit value from a register.
 * @param reg The register address to read from.
 * @return uint16_t The value read from the register.
 * @details This function reads a 16-bit value from a register on the I2C
 * device.
 */
uint16_t I2CController::readRegister(uint8_t reg) {
  if (write(i2c_fd_, &reg, 1) != 1) {
	perror("I2C write failed");
  }
  std::array<uint8_t, 2> buffer;
  if (read(i2c_fd_, buffer.data(), buffer.size()) != 2) {
	perror("I2C read failed");
  }
  return (buffer[0] << 8) | buffer[1];
}
