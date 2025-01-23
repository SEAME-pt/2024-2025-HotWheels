#include "EngineController.hpp"
#include <QDebug>
#include <atomic>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

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

EngineController::EngineController(int servo_addr, int motor_addr, QObject *parent)
    : QObject(parent)
    , servo_addr_(servo_addr)
    , motor_addr_(motor_addr)
    , m_running(false)
    , m_current_speed(0)
    , m_current_angle(0)
{
    pcontrol = new PeripheralController();

    // Initialize I2C buses
    servo_bus_fd_ = open("/dev/i2c-1", O_RDWR);
    motor_bus_fd_ = open("/dev/i2c-1", O_RDWR);

    if (servo_bus_fd_ < 0 || motor_bus_fd_ < 0) {
        perror("Failed to open I2C device");
        disable();
        return;
    }

    // Set device addresses
    if (ioctl(servo_bus_fd_, I2C_SLAVE, servo_addr_) < 0
        || ioctl(motor_bus_fd_, I2C_SLAVE, motor_addr_) < 0) {
        perror("Failed to set I2C address");
        disable();
        return;
    }

    pcontrol->init_servo(servo_bus_fd_, isDisabled());
    pcontrol->init_motors(motor_bus_fd_, isDisabled());
}

EngineController::~EngineController()
{
    if (!isDisabled()) {
        stop();
        close(servo_bus_fd_);
        close(motor_bus_fd_);
    }
    delete pcontrol;
}

void EngineController::start()
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot start." << std::endl;
        return;
    }
    m_running = true;
}

void EngineController::stop()
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot stop." << std::endl;
        return;
    }

    m_running = false;
    set_speed(0);
    set_steering(0);
}

void EngineController::disable()
{
    m_disabled = true;
    m_running = false;
    m_current_speed = 0;
    m_current_angle = 0;
}

bool EngineController::isDisabled() const
{
    return m_disabled;
}

void EngineController::setDirection(CarDirection newDirection)
{
    if (newDirection != this->m_currentDirection) {
        emit this->directionUpdated(newDirection);
        this->m_currentDirection = newDirection;
    }
}

void EngineController::set_speed(int speed)
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot set speed." << std::endl;
        return;
    }

    speed = clamp(speed, -100, 100);
    int pwm_value = static_cast<int>(std::abs(speed) / 100.0 * 4096);

    if (speed > 0) { // Forward (but actually backward because joysticks are reversed)
        pcontrol->set_motor_pwm(0, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(1, 0, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(2, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(5, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(6, 0, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(7, pwm_value, motor_bus_fd_, isDisabled());
        setDirection(CarDirection::Reverse);
    } else if (speed < 0) { // Backwards
        pcontrol->set_motor_pwm(0, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(1, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(2, 0, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(5, 0, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(6, pwm_value, motor_bus_fd_, isDisabled());
        pcontrol->set_motor_pwm(7, pwm_value, motor_bus_fd_, isDisabled());
        setDirection(CarDirection::Drive);
    } else { // Stop
        for (int channel = 0; channel < 9; ++channel)
            pcontrol->set_motor_pwm(channel, 0, motor_bus_fd_, isDisabled());
        setDirection(CarDirection::Stop);
    }
    m_current_speed = speed;
}

void EngineController::set_steering(int angle)
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot set steering angle." << std::endl;
        return;
    }

    angle = clamp(angle, -MAX_ANGLE, MAX_ANGLE);
    int pwm = 0;
    if (angle < 0) {
        pwm = SERVO_CENTER_PWM
              + static_cast<int>((angle / static_cast<float>(MAX_ANGLE))
                                 * (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
    } else if (angle > 0) {
        pwm = SERVO_CENTER_PWM
              + static_cast<int>((angle / static_cast<float>(MAX_ANGLE))
                                 * (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
    } else {
        pwm = SERVO_CENTER_PWM;
    }

    pcontrol->set_servo_pwm(STEERING_CHANNEL, 0, pwm, servo_bus_fd_, isDisabled());
    m_current_angle = angle;
    emit this->steeringUpdated(angle);
}

/* void EngineController::init_servo()
{
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
} */

/* void EngineController::init_motors()
{
    pcontrol->write_byte_data(motor_bus_fd_, 0x00, 0x20);

    int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
    int oldmode = read_byte_data(motor_bus_fd_, 0x00);
    int newmode = (oldmode & 0x7F) | 0x10;

    pcontrol->write_byte_data(motor_bus_fd_, 0x00, newmode);
    pcontrol->write_byte_data(motor_bus_fd_, 0xFE, prescale);
    pcontrol->write_byte_data(motor_bus_fd_, 0x00, oldmode);
    usleep(5000);
    pcontrol->write_byte_data(motor_bus_fd_, 0x00, oldmode | 0xa1);
} */

/* void EngineController::set_servo_pwm(int channel, int on_value, int off_value)
{
    int base_reg = 0x06 + (channel * 4);
    write_byte_data(servo_bus_fd_, base_reg, on_value & 0xFF);
    write_byte_data(servo_bus_fd_, base_reg + 1, on_value >> 8);
    write_byte_data(servo_bus_fd_, base_reg + 2, off_value & 0xFF);
    write_byte_data(servo_bus_fd_, base_reg + 3, off_value >> 8);
} */

/* void EngineController::set_motor_pwm(int channel, int value)
{
    value = clamp(value, 0, 4096);
    int base_reg = 0x06 + (4 * channel);
    // qDebug() << "Set motor pwm to " << value << "in channel " << channel;
    write_byte_data(motor_bus_fd_, base_reg, value & 0xFF);
    write_byte_data(motor_bus_fd_, base_reg + 1, value >> 8);
} */

/* void EngineController::write_byte_data(int fd, int reg, int value)
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot write byte data." << std::endl;
        return;
    }

    if (i2c_smbus_write_byte_data(fd, reg, value) < 0) {
        throw std::runtime_error("I2C write failed");
    }
} */

/* int EngineController::read_byte_data(int fd, int reg)
{
    if (isDisabled()) {
        std::cerr << "EngineController is disabled. Cannot read byte data." << std::endl;
        return 0;
    }

    int result = i2c_smbus_read_byte_data(fd, reg);
    if (result < 0) {
        throw std::runtime_error("I2C read failed");
    }
    return result;
} */
