#ifndef MOCKPERIPHERALCONTROLLER_HPP
#define MOCKPERIPHERALCONTROLLER_HPP

#include "IPeripheralController.hpp"
#include <gmock/gmock.h>

class MockPeripheralController : public IPeripheralController
{
public:
    MOCK_METHOD(int, i2c_smbus_write_byte_data, (int file, uint8_t command, uint8_t value), (override));
    MOCK_METHOD(int, i2c_smbus_read_byte_data, (int file, uint8_t command), (override));

    MOCK_METHOD(void, write_byte_data, (int fd, int reg, int value), (override));
    MOCK_METHOD(int, read_byte_data, (int fd, int reg), (override));

    MOCK_METHOD(void, set_servo_pwm, (int channel, int on_value, int off_value), (override));
    MOCK_METHOD(void, set_motor_pwm, (int channel, int value), (override));

    MOCK_METHOD(void, init_servo, (), (override));
    MOCK_METHOD(void, init_motors, (), (override));
};

#endif // MOCKPERIPHERALCONTROLLER_HPP
