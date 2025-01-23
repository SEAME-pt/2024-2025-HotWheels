#include <gtest/gtest.h>
#include <gmock/gmock.h> // Include for Google Mock
#include "MockEngine.hpp" // Include your MockEngineController header
#include <stdexcept>

// Test fixture class for EngineController

class EngineControllerTest : public ::testing::Test {
    protected:
        MockEngineController* mockEngineController;
        I2CMock& i2c_mock = I2CMock::getInstance();

    void SetUp() override {
        mockEngineController = new MockEngineController();

        // Set default expectations for I2C operations
        EXPECT_CALL(i2c_mock, open(testing::_, testing::_))
            .WillRepeatedly(testing::Return(1));  // Return valid file descriptor

        EXPECT_CALL(i2c_mock, ioctl(testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Return(0));  // Return success

        EXPECT_CALL(i2c_mock, i2c_smbus_write_byte_data(testing::_, testing::_, testing::_))
            .WillRepeatedly(testing::Return(0));  // Return success

        EXPECT_CALL(i2c_mock, i2c_smbus_read_byte_data(testing::_, testing::_))
            .WillRepeatedly(testing::Return(0));  // Return success
    }

    void TearDown() override {
        delete mockEngineController;
    }
};

// Test start method
TEST_F(EngineControllerTest, Start) {
    // Mock behavior for start method
    EXPECT_CALL(*mockEngineController, start()).Times(1);
    mockEngineController->start();
}

// Test stop method
TEST_F(EngineControllerTest, Stop) {
    // Mock behavior for stop method
    EXPECT_CALL(*mockEngineController, stop()).Times(1);
    mockEngineController->stop();
}

// Test set_speed method
TEST_F(EngineControllerTest, SetSpeed) {
    EXPECT_CALL(*mockEngineController, set_speed(50)).Times(1);
    mockEngineController->set_speed(50);

    EXPECT_CALL(*mockEngineController, set_speed(-50)).Times(1);
    mockEngineController->set_speed(-50);

    EXPECT_CALL(*mockEngineController, set_speed(150)).Times(1);
    mockEngineController->set_speed(150);

    EXPECT_CALL(*mockEngineController, set_speed(-150)).Times(1);
    mockEngineController->set_speed(-150);
}

// Test set_steering method
TEST_F(EngineControllerTest, SetSteering) {
    EXPECT_CALL(*mockEngineController, set_steering(30)).Times(1);
    mockEngineController->set_steering(30);

    EXPECT_CALL(*mockEngineController, set_steering(-30)).Times(1);
    mockEngineController->set_steering(-30);

    EXPECT_CALL(*mockEngineController, set_steering(150)).Times(1);
    mockEngineController->set_steering(150);

    EXPECT_CALL(*mockEngineController, set_steering(-150)).Times(1);
    mockEngineController->set_steering(-150);
}

// Test init_servo method
/* TEST_F(EngineControllerTest, InitServo) {
    EXPECT_CALL(*mockEngineController, init_servo()).Times(1);
    mockEngineController->init_servo();
} */

// Test init_motors method
/* TEST_F(EngineControllerTest, InitMotors) {
    EXPECT_CALL(*mockEngineController, init_motors()).Times(1);
    mockEngineController->init_motors();
} */

// Test write_byte_data method
/* TEST_F(EngineControllerTest, WriteByteData) {
    int fd = 0;  // Dummy FD
    EXPECT_CALL(*mockEngineController, write_byte_data(fd, 0x00, 0x06)).Times(1);
    mockEngineController->write_byte_data(fd, 0x00, 0x06);
} */


// Test read_byte_data method
/* TEST_F(EngineControllerTest, ReadByteData) {
    int fd = 0;  // Dummy FD
    EXPECT_CALL(*mockEngineController, read_byte_data(fd, 0x00)).Times(1).WillOnce(testing::Return(42)); // Example mocked return value
    int value = mockEngineController->read_byte_data(fd, 0x00);
    EXPECT_EQ(value, 42);
} */

// Test process_joystick method
/* TEST_F(EngineControllerTest, ProcessJoystick) {
    EXPECT_CALL(*mockEngineController, process_joystick()).Times(1);
    mockEngineController->process_joystick();
} */

// Test set_servo_pwm method
/* TEST_F(EngineControllerTest, SetServoPwm) {
    EXPECT_CALL(*mockEngineController, set_servo_pwm(0, 0, 2048)).Times(1);
    mockEngineController->set_servo_pwm(0, 0, 2048);
} */

// Test set_motor_pwm method
/* TEST_F(EngineControllerTest, SetMotorPwm) {
    EXPECT_CALL(*mockEngineController, set_motor_pwm(0, 2048)).Times(1);
    mockEngineController->set_motor_pwm(0, 2048);
} */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
