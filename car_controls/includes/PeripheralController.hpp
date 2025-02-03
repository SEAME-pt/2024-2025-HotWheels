#ifndef PERIPHERALCONTROLLER_HPP
#define PERIPHERALCONTROLLER_HPP

#include "IPeripheralController.hpp"
#include <QDebug>
#include <QObject>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

class PeripheralController : public IPeripheralController {
private:
  int servo_bus_fd_;
  int motor_bus_fd_;
  int servo_addr_;
  int motor_addr_;

public:
  /*!
   * Constructor for the PeripheralController class, initializing the I2C
   * communication for the servo and motor controllers.
   *
   * @param servo_addr The I2C address for the servo controller.
   * @param motor_addr The I2C address for the motor controller.
   */
  PeripheralController(int servo_addr, int motor_addr);

  /*!
   * Destructor for the PeripheralController class, closing the I2C
   * communication channels.
   */
  ~PeripheralController() override;

  /*!
   * Writes a byte of data to the specified I2C device.
   *
   * @param file The I2C file descriptor.
   * @param command The command byte for the I2C device.
   * @param value The value byte to write to the I2C device.
   * @return The result of the I2C operation.
   */
  int i2c_smbus_write_byte_data(int file, uint8_t command,
                                uint8_t value) override;

  /*!
   * Reads a byte of data from the specified I2C device.
   *
   * @param file The I2C file descriptor.
   * @param command The command byte to request data from the I2C device.
   * @return The byte read from the I2C device.
   */
  int i2c_smbus_read_byte_data(int file, uint8_t command) override;

  /*!
   * Writes a byte of data to a specific register on the I2C device.
   *
   * @param fd The file descriptor for the I2C device.
   * @param reg The register to write to.
   * @param value The value to write to the register.
   */
  virtual void write_byte_data(int fd, int reg, int value) override;

  /*!
   * Reads a byte of data from a specific register on the I2C device.
   *
   * @param fd The file descriptor for the I2C device.
   * @param reg The register to read from.
   * @return The byte read from the register.
   */
  virtual int read_byte_data(int fd, int reg) override;

  /*!
   * Sets the PWM value for the servo motor.
   *
   * @param channel The servo channel to control.
   * @param on_value The value to set when the PWM is "on".
   * @param off_value The value to set when the PWM is "off".
   */
  void set_servo_pwm(int channel, int on_value, int off_value) override;

  /*!
   * Sets the PWM value for the motor.
   *
   * @param channel The motor channel to control.
   * @param value The PWM value to set.
   */
  void set_motor_pwm(int channel, int value) override;

  /*!
   * Initializes the servo motor by sending necessary commands over I2C.
   */
  void init_servo() override;

  /*!
   * Initializes the motor controller by setting the correct frequency and mode.
   */
  void init_motors() override;
};

#endif // PERIPHERALCONTROLLER_HPP
