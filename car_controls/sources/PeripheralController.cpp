#include "PeripheralController.hpp"

/**
 * Represents data formats for I2C SMBus communication:
 * - `byte`: Single byte of data.
 * - `word`: 16-bit (2 byte) data.
 * - `block`: Array for up to 34-byte block transfers.
 */
union i2c_smbus_data {
  uint8_t byte;
  uint16_t word;
  uint8_t block[34]; // Block size for SMBus
};

/* ------------------------------------ */

#define I2C_SMBUS_WRITE 0
#define I2C_SMBUS_READ 1
#define I2C_SMBUS_BYTE_DATA 2

template <typename T> T clamp(T value, T min_val, T max_val) {
  return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

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

PeripheralController::~PeripheralController() {
  close(servo_bus_fd_);
  close(motor_bus_fd_);
}

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

void PeripheralController::write_byte_data(int fd, int reg, int value) {
  if (i2c_smbus_write_byte_data(fd, reg, value) < 0) {
    throw std::runtime_error("I2C write failed");
  }
}

int PeripheralController::read_byte_data(int fd, int reg) {
  int result = i2c_smbus_read_byte_data(fd, reg);
  if (result < 0) {
    throw std::runtime_error("I2C read failed");
  }
  return result;
}

void PeripheralController::set_servo_pwm(int channel, int on_value,
                                         int off_value) {
  int base_reg = 0x06 + (channel * 4);
  write_byte_data(servo_bus_fd_, base_reg, on_value & 0xFF);
  write_byte_data(servo_bus_fd_, base_reg + 1, on_value >> 8);
  write_byte_data(servo_bus_fd_, base_reg + 2, off_value & 0xFF);
  write_byte_data(servo_bus_fd_, base_reg + 3, off_value >> 8);
}

void PeripheralController::set_motor_pwm(int channel, int value) {
  value = clamp(value, 0, 4095);
  write_byte_data(motor_bus_fd_, 0x06 + (4 * channel), 0);
  write_byte_data(motor_bus_fd_, 0x07 + (4 * channel), 0);
  write_byte_data(motor_bus_fd_, 0x08 + (4 * channel), value & 0xFF);
  write_byte_data(motor_bus_fd_, 0x09 + (4 * channel), value >> 8);
}

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
