#include "MockPeripheralController.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

TEST(PeripheralControllerTest, TestServoPWM)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, set_servo_pwm(0, 1024, 2048)).Times(1);
    EXPECT_CALL(mockController, set_servo_pwm(1, 0, 4096)).Times(1);

    mockController.set_servo_pwm(0, 1024, 2048);
    mockController.set_servo_pwm(1, 0, 4096);
}

TEST(PeripheralControllerTest, TestMotorPWM)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, set_motor_pwm(0, 1500)).Times(1);
    EXPECT_CALL(mockController, set_motor_pwm(1, 3000)).Times(1);

    mockController.set_motor_pwm(0, 1500);
    mockController.set_motor_pwm(1, 3000);
}

TEST(PeripheralControllerTest, TestInitServo)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, init_servo()).Times(1);

    mockController.init_servo();
}

TEST(PeripheralControllerTest, TestInitMotors)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, init_motors()).Times(1);

    mockController.init_motors();
}

TEST(PeripheralControllerTest, TestI2CWriteByteData)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, i2c_smbus_write_byte_data(1, 0x10, 0x20)).WillOnce(Return(0));

    int result = mockController.i2c_smbus_write_byte_data(1, 0x10, 0x20);

    EXPECT_EQ(result, 0);
}

TEST(PeripheralControllerTest, TestI2CReadByteData)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, i2c_smbus_read_byte_data(1, 0x10)).WillOnce(Return(0x30));

    int result = mockController.i2c_smbus_read_byte_data(1, 0x10);

    EXPECT_EQ(result, 0x30);
}

TEST(PeripheralControllerTest, TestWriteByteDataException)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, write_byte_data(_, _, _)).WillOnce(Throw(std::runtime_error("I2C write failed")));

    EXPECT_THROW(mockController.write_byte_data(1, 0x10, 0x20), std::runtime_error);
}

TEST(PeripheralControllerTest, TestReadByteDataException)
{
    MockPeripheralController mockController;

    EXPECT_CALL(mockController, read_byte_data(_, _)).WillOnce(Throw(std::runtime_error("I2C read failed")));

    EXPECT_THROW(mockController.read_byte_data(1, 0x10), std::runtime_error);
}
