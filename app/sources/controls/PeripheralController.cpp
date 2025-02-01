/**
 * @file PeripheralController.cpp
 * @brief Implementation of the PeripheralController class for I2C
 * communication.
 * @version 0.1
 * @date 2025-01-31
 * @details This class is responsible for handling I2C communication with the
 * servo and motor controllers.
 * @note  This class uses the Linux I2C SMBus interface for communication.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @warning Ensure that the I2C devices are properly connected and configured on
 * your system.
 *
 * @see PeripheralController.hpp for the class definition.
 *
 * @copyright Copyright (c) 2025
 */

#include "PeripheralController.hpp"

/**
 * @union i2c_smbus_data that contains the data to be sent or received.
 * @brief Union that contains the data to be sent or received over I2C.
 */
union i2c_smbus_data {
  uint8_t byte;
  uint16_t word;
  uint8_t block[34]; // Block size for SMBus
};

/* ------------------------------------ */
/** @def I2C_SMBUS_WRITE Macro to indicate a write operation. */
#define I2C_SMBUS_WRITE 0
/** @def I2C_SMBUS_READ Macro to indicate a read operation. */
#define I2C_SMBUS_READ 1
/** @def I2C_SMBUS_BYTE_DATA Macro to indicate a byte data operation. */
#define I2C_SMBUS_BYTE_DATA 2

/**
 * @brief Clamp a value to a specified range.
 *
 * @tparam T
 * @param value
 * @param min_val
 * @param max_val
 * @return T
 */
template <typename T> T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/**
 * @brief Constructor for the PeripheralController class.
 * @param servo_addr I2C address of the servo controller.
 * @param motor_addr I2C address of the motor controller.
 * @throws std::runtime_error if the I2C device cannot be opened or the address
 * cannot be set.
 * @details Initializes the I2C buses and sets the device addresses for the
 * servo and motor controllers.
 */
PeripheralController::PeripheralController(int servo_addr, int motor_addr)
    : servo_addr_(servo_addr), motor_addr_(motor_addr) {
  // Initialize I2C buses
  servo_bus_fd_ = open("/dev/i2c-1", O_RDWR);
  motor_bus_fd_ = open("/dev/i2c-1", O_RDWR);

  if (servo_bus_fd_ < 0 || motor_bus_fd_ < 0) {
    throw std::runtime_error("Failed to open I2C device");
    return;
  }

  // Set device addresses
  if (ioctl(servo_bus_fd_, I2C_SLAVE, servo_addr_) < 0 ||
      ioctl(motor_bus_fd_, I2C_SLAVE, motor_addr_) < 0) {
    throw std::runtime_error("Failed to set I2C address");
    return;
  }
}

/**
 * @brief Destructor for the PeripheralController class.
 * @details Closes the I2C file descriptors for the servo and motor controllers.
 */
PeripheralController::~PeripheralController() {
  close(servo_bus_fd_);
  close(motor_bus_fd_);
}

/**
 * @brief Writes a byte of data to a specific register on the I2C device.
 * @param file File descriptor of the I2C device.
 * @param command Register address to write to.
 * @param value Byte value to write.
 * @return int 0 on success, -1 on failure.
 */
int PeripheralController::i2c_smbus_write_byte_data(int file, uint8_t command,
                                                    uint8_t value) {
  union i2c_smbus_data data;
  data.byte = value;

  struct i2c_smbus_ioctl_data args;
  args.read_write = I2C_SMBUS_WRITE;
  args.command = command;
  args.size = I2C_SMBUS_BYTE_DATA;
  args.data = &data;

  return ioctl(file, I2C_SMBUS, &args);
}

/**
 * @brief Reads a byte of data from a specific register on the I2C device.
 * @param file File descriptor of the I2C device.
 * @param command Register address to read from.
 * @return int Byte value read, or -1 on failure.
 */
int PeripheralController::i2c_smbus_read_byte_data(int file, uint8_t command) {
  union i2c_smbus_data data;

  struct i2c_smbus_ioctl_data args;
  args.read_write = I2C_SMBUS_READ;
  args.command = command;
  args.size = I2C_SMBUS_BYTE_DATA;
  args.data = &data;

  if (ioctl(file, I2C_SMBUS, &args) < 0) {
    return -1;
  }
  return data.byte;
}

/**
 * @brief Writes a byte of data to a specific register.
 * @param fd File descriptor of the I2C device.
 * @param reg Register address to write to.
 * @param value Byte value to write.
 * @throws std::runtime_error if the write operation fails.
 */
void PeripheralController::write_byte_data(int fd, int reg, int value) {
  if (i2c_smbus_write_byte_data(fd, reg, value) < 0) {
    throw std::runtime_error("I2C write failed");
  }
}

/**
 * @brief Reads a byte of data from a specific register.
 * @param fd File descriptor of the I2C device.
 * @param reg Register address to read from.
 * @return int Byte value read.
 * @throws std::runtime_error if the read operation fails.
 */
int PeripheralController::read_byte_data(int fd, int reg) {
  int result = i2c_smbus_read_byte_data(fd, reg);
  if (result < 0) {
    throw std::runtime_error("I2C read failed");
  }
  return result;
}

/**
 * @brief Sets the PWM values for a specific servo channel.
 * @param channel Servo channel number.
 * @param on_value PWM on value.
 * @param off_value PWM off value.
 * @details Writes the on and off values to the appropriate registers for the
 * specified servo channel.
 */
void PeripheralController::set_servo_pwm(int channel, int on_value,
                                         int off_value) {
  int base_reg = 0x06 + (channel * 4);
  write_byte_data(servo_bus_fd_, base_reg, on_value & 0xFF);
  write_byte_data(servo_bus_fd_, base_reg + 1, on_value >> 8);
  write_byte_data(servo_bus_fd_, base_reg + 2, off_value & 0xFF);
  write_byte_data(servo_bus_fd_, base_reg + 3, off_value >> 8);
}

/**
 * @brief Sets the PWM value for a specific motor channel.
 * @param channel Motor channel number.
 * @param value PWM value (0-4095).
 * @details Clamps the value to the range [0, 4095] and writes it to the
 * appropriate registers for the specified motor channel.
 */
void PeripheralController::set_motor_pwm(int channel, int value) {
  value = clamp(value, 0, 4095);
  write_byte_data(motor_bus_fd_, 0x06 + (4 * channel), 0);
  write_byte_data(motor_bus_fd_, 0x07 + (4 * channel), 0);
  write_byte_data(motor_bus_fd_, 0x08 + (4 * channel), value & 0xFF);
  write_byte_data(motor_bus_fd_, 0x09 + (4 * channel), value >> 8);
}

/**
 * @brief Initializes the servo controller.
 * @details Configures the servo controller with the necessary settings and
 * delays.
 */
void PeripheralController::init_servo() {
  write_byte_data(servo_bus_fd_, 0x00, 0x06);
  usleep(100000);

  write_byte_data(servo_bus_fd_, 0x00, 0x10);
  usleep(100000);

  write_byte_data(servo_bus_fd_, 0xFE, 0x79);
  usleep(100000);

  write_byte_data(servo_bus_fd_, 0x01, 0x04);
  usleep(100000);

  write_byte_data(servo_bus_fd_, 0x00, 0x20);
  usleep(100000);
}

/**
 * @brief Initializes the motor controller.
 * @details Configures the motor controller with the necessary settings and
 * delays.
 */
void PeripheralController::init_motors() {
  write_byte_data(motor_bus_fd_, 0x00, 0x20);

  int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
  int oldmode = read_byte_data(motor_bus_fd_, 0x00);
  int newmode = (oldmode & 0x7F) | 0x10;

  write_byte_data(motor_bus_fd_, 0x00, newmode);
  write_byte_data(motor_bus_fd_, 0xFE, prescale);
  write_byte_data(motor_bus_fd_, 0x00, oldmode);
  usleep(5000);
  write_byte_data(motor_bus_fd_, 0x00, oldmode | 0xa1);
}
