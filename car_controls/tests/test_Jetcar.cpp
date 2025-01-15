#include <gtest/gtest.h>
#include <gmock/gmock.h> // Include for Google Mock
#include "MockJetcar.hpp" // Include your MockJetcar header
#include <stdexcept>

// Test fixture class for Jetcar
class JetcarTest : public ::testing::Test {
protected:
	MockJetcar* mockJetcar;

	void SetUp() override {
		mockJetcar = new MockJetcar();
	}

	void TearDown() override {
		delete mockJetcar;
	}
};


// Test constructor with default addresses
TEST_F(JetcarTest, ConstructorDefaultAddresses) {
	EXPECT_NO_THROW(Jetcar());
}

// Test constructor with custom addresses
TEST_F(JetcarTest, ConstructorCustomAddresses) {
	EXPECT_NO_THROW(Jetcar(0x40, 0x60));
}

// Test start method
TEST_F(JetcarTest, Start) {
	// Mock behavior for start method
	EXPECT_CALL(*mockJetcar, start()).Times(1);
	mockJetcar->start();
}

// Test stop method
TEST_F(JetcarTest, Stop) {
	// Mock behavior for stop method
	EXPECT_CALL(*mockJetcar, stop()).Times(1);
	mockJetcar->stop();
}

// Test set_speed method
TEST_F(JetcarTest, SetSpeed) {
	EXPECT_CALL(*mockJetcar, set_speed(50)).Times(1);
	mockJetcar->set_speed(50);

	EXPECT_CALL(*mockJetcar, set_speed(-50)).Times(1);
	mockJetcar->set_speed(-50);

	EXPECT_CALL(*mockJetcar, set_speed(150)).Times(1);
	mockJetcar->set_speed(150);

	EXPECT_CALL(*mockJetcar, set_speed(-150)).Times(1);
	mockJetcar->set_speed(-150);
}

// Test set_steering method
TEST_F(JetcarTest, SetSteering) {
	EXPECT_CALL(*mockJetcar, set_steering(30)).Times(1);
	mockJetcar->set_steering(30);

	EXPECT_CALL(*mockJetcar, set_steering(-30)).Times(1);
	mockJetcar->set_steering(-30);

	EXPECT_CALL(*mockJetcar, set_steering(150)).Times(1);
	mockJetcar->set_steering(150);

	EXPECT_CALL(*mockJetcar, set_steering(-150)).Times(1);
	mockJetcar->set_steering(-150);
}

// Test init_servo method
TEST_F(JetcarTest, InitServo) {
	EXPECT_CALL(*mockJetcar, init_servo()).Times(1);
	mockJetcar->init_servo();
}

// Test init_motors method
TEST_F(JetcarTest, InitMotors) {
	EXPECT_CALL(*mockJetcar, init_motors()).Times(1);
	mockJetcar->init_motors();
}

// Test write_byte_data method
TEST_F(JetcarTest, WriteByteData) {
	int fd = 0;  // Dummy FD
	EXPECT_CALL(*mockJetcar, write_byte_data(fd, 0x00, 0x06)).Times(1);
	mockJetcar->write_byte_data(fd, 0x00, 0x06);
}


// Test read_byte_data method
TEST_F(JetcarTest, ReadByteData) {
	int fd = 0;  // Dummy FD
	EXPECT_CALL(*mockJetcar, read_byte_data(fd, 0x00)).Times(1).WillOnce(testing::Return(42)); // Example mocked return value
	int value = mockJetcar->read_byte_data(fd, 0x00);
	EXPECT_EQ(value, 42);
}

// Test process_joystick method
TEST_F(JetcarTest, ProcessJoystick) {
	EXPECT_CALL(*mockJetcar, process_joystick()).Times(1);
	mockJetcar->process_joystick();
}

// Test set_servo_pwm method
TEST_F(JetcarTest, SetServoPwm) {
	EXPECT_CALL(*mockJetcar, set_servo_pwm(0, 0, 2048)).Times(1);
	mockJetcar->set_servo_pwm(0, 0, 2048);
}

// Test set_motor_pwm method
TEST_F(JetcarTest, SetMotorPwm) {
	EXPECT_CALL(*mockJetcar, set_motor_pwm(0, 2048)).Times(1);
	mockJetcar->set_motor_pwm(0, 2048);
}

// Test clamp function
TEST(ClampTest, ClampFunction) {
	EXPECT_EQ(std::clamp(5, 1, 10), 5);
	EXPECT_EQ(std::clamp(0, 1, 10), 1);
	EXPECT_EQ(std::clamp(15, 1, 10), 10);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
