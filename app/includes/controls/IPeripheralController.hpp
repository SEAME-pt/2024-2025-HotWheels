/**
 * @file IPeripheralController.hpp
 * @brief
 * @version 0.1
 * @date 2025-01-31
 * @details
 * @note
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef IPERIPHERALCONTROLLER_HPP
#define IPERIPHERALCONTROLLER_HPP

#include <QDebug>
#include <QObject>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

class IPeripheralController {
public:
  virtual ~IPeripheralController() = default;

  virtual int i2c_smbus_write_byte_data(int file, uint8_t command,
                                        uint8_t value) = 0;
  virtual int i2c_smbus_read_byte_data(int file, uint8_t command) = 0;

  virtual void write_byte_data(int fd, int reg, int value) = 0;
  virtual int read_byte_data(int fd, int reg) = 0;

  virtual void set_servo_pwm(int channel, int on_value, int off_value) = 0;
  virtual void set_motor_pwm(int channel, int value) = 0;

  virtual void init_servo() = 0;
  virtual void init_motors() = 0;
};

#endif
