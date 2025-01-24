#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "EngineController.hpp"

// Mock for i2c functions
class I2CMock {
public:
    static I2CMock& getInstance() {
        static I2CMock instance;
        return instance;
    }

    MOCK_METHOD(int, i2c_smbus_write_byte_data, (int file, uint8_t reg, uint8_t value));
    MOCK_METHOD(int, i2c_smbus_read_byte_data, (int file, uint8_t reg));
    MOCK_METHOD(int, open, (const char* pathname, int flags));
    MOCK_METHOD(int, ioctl, (int fd, unsigned long request, void* argp));
};

// Function overrides that redirect to our mock
extern "C" {
    int i2c_smbus_write_byte_data(int file, uint8_t reg, uint8_t value) {
        return I2CMock::getInstance().i2c_smbus_write_byte_data(file, reg, value);
    }

    int i2c_smbus_read_byte_data(int file, uint8_t reg) {
        return I2CMock::getInstance().i2c_smbus_read_byte_data(file, reg);
    }

    int open(const char* pathname, int flags, ...) {
        return I2CMock::getInstance().open(pathname, flags);
    }

    int ioctl(int fd, unsigned long request, ...) {
        va_list args;
        va_start(args, request);
        void* argp = va_arg(args, void*);
        va_end(args);
        return I2CMock::getInstance().ioctl(fd, request, argp);
    }
}

// I2C interface for Jetcar
class I2CInterface {
public:
    virtual ~I2CInterface() = default;
    virtual int write_byte_data(int fd, uint8_t reg, uint8_t value) = 0;
    virtual int read_byte_data(int fd, uint8_t reg) = 0;
};

// Mock I2C implementation
class MockI2C : public I2CInterface {
public:
    MOCK_METHOD(int, write_byte_data, (int fd, uint8_t reg, uint8_t value), (override));
    MOCK_METHOD(int, read_byte_data, (int fd, uint8_t reg), (override));
};

class MockEngineController : public EngineController
{
    public:
        explicit MockEngineController(std::unique_ptr<I2CInterface> i2c = std::make_unique<MockI2C>())
        : i2c_(std::move(i2c)) {}

        MOCK_METHOD(void, start, ());
        MOCK_METHOD(void, stop, ());
        MOCK_METHOD(void, set_speed, (int speed));
        MOCK_METHOD(void, set_steering, (int angle));
        /* MOCK_METHOD(void, init_servo, ());
        MOCK_METHOD(void, init_motors, ());
        MOCK_METHOD(void, write_byte_data, (int fd, int reg, int value));
        MOCK_METHOD(int, read_byte_data, (int fd, int reg));
        MOCK_METHOD(void, process_joystick, ());
        MOCK_METHOD(void, set_servo_pwm, (int channel, int on, int off));
        MOCK_METHOD(void, set_motor_pwm, (int channel, int pwm)); */

        /* int write_byte_data(int fd, uint8_t reg, uint8_t value) {
            return i2c_->write_byte_data(fd, reg, value);
        }

        int read_byte_data(int fd, uint8_t reg) {
            return i2c_->read_byte_data(fd, reg);
        } */

    private:
        std::unique_ptr<I2CInterface> i2c_;
};
