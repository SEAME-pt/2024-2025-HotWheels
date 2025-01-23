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

int i2c_smbus_write_byte_data(int file, uint8_t command, uint8_t value)
{
    union i2c_smbus_data data;
    data.byte = value;

    struct i2c_smbus_ioctl_data args;
    args.read_write = I2C_SMBUS_WRITE;
    args.command = command;
    args.size = I2C_SMBUS_BYTE_DATA;
    args.data = &data;

    return ioctl(file, I2C_SMBUS, &args);
}

int i2c_smbus_read_byte_data(int file, uint8_t command)
{
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

template<typename T>
T clamp(T value, T min_val, T max_val)
{
    return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

PeripheralController::PeripheralController(QObject *parent) : QObject(parent) {}

PeripheralController::~PeripheralController() {}

void PeripheralController::write_byte_data(int fd, int reg, int value, bool disabled)
{
    if (disabled) {
        std::cerr << "EngineController is disabled. Cannot write byte data." << std::endl;
        return;
    }

    if (i2c_smbus_write_byte_data(fd, reg, value) < 0) {
        throw std::runtime_error("I2C write failed");
    }
}

int PeripheralController::read_byte_data(int fd, int reg, bool disabled)
{
    if (disabled) {
        std::cerr << "EngineController is disabled. Cannot read byte data." << std::endl;
        return 0;
    }

    int result = i2c_smbus_read_byte_data(fd, reg);
    if (result < 0) {
        throw std::runtime_error("I2C read failed");
    }
    return result;
}

void PeripheralController::set_servo_pwm(int channel, int on_value, int off_value, int servo_bus_fd_, bool disabled)
{
    int base_reg = 0x06 + (channel * 4);
    write_byte_data(servo_bus_fd_, base_reg, on_value & 0xFF, disabled);
    write_byte_data(servo_bus_fd_, base_reg + 1, on_value >> 8, disabled);
    write_byte_data(servo_bus_fd_, base_reg + 2, off_value & 0xFF, disabled);
    write_byte_data(servo_bus_fd_, base_reg + 3, off_value >> 8, disabled);
}

void PeripheralController::set_motor_pwm(int channel, int value, int motor_bus_fd_, bool disabled)
{
    value = clamp(value, 0, 4096);
    int base_reg = 0x06 + (4 * channel);
    // qDebug() << "Set motor pwm to " << value << "in channel " << channel;
    write_byte_data(motor_bus_fd_, base_reg, value & 0xFF, disabled);
    write_byte_data(motor_bus_fd_, base_reg + 1, value >> 8, disabled);
}

void PeripheralController::init_servo(int servo_bus_fd_, bool disabled)
{
    write_byte_data(servo_bus_fd_, 0x00, 0x06, disabled);
    usleep(100000);

    write_byte_data(servo_bus_fd_, 0x00, 0x10, disabled);
    usleep(100000);

    write_byte_data(servo_bus_fd_, 0xFE, 0x79, disabled);
    usleep(100000);

    write_byte_data(servo_bus_fd_, 0x01, 0x04, disabled);
    usleep(100000);

    write_byte_data(servo_bus_fd_, 0x00, 0x20, disabled);
    usleep(100000);
}

void PeripheralController::init_motors(int motor_bus_fd_, bool disabled)
{
    write_byte_data(motor_bus_fd_, 0x00, 0x20, disabled);

    int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
    int oldmode = read_byte_data(motor_bus_fd_, 0x00, disabled);
    int newmode = (oldmode & 0x7F) | 0x10;

    write_byte_data(motor_bus_fd_, 0x00, newmode, disabled);
    write_byte_data(motor_bus_fd_, 0xFE, prescale, disabled);
    write_byte_data(motor_bus_fd_, 0x00, oldmode, disabled);
    usleep(5000);
    write_byte_data(motor_bus_fd_, 0x00, oldmode | 0xa1, disabled);
}
