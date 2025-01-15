#include <gmock/gmock.h>
#include "../Jetcar.hpp"

class MockJetcar : public Jetcar
{
	public:
		MOCK_METHOD(void, start, ());
		MOCK_METHOD(void, stop, ());
		MOCK_METHOD(void, set_speed, (int speed));
		MOCK_METHOD(void, set_steering, (int angle));
		MOCK_METHOD(void, init_servo, ());
		MOCK_METHOD(void, init_motors, ());
		MOCK_METHOD(void, write_byte_data, (int fd, int reg, int value));
		MOCK_METHOD(int, read_byte_data, (int fd, int reg));
		MOCK_METHOD(void, process_joystick, ());
		MOCK_METHOD(void, set_servo_pwm, (int channel, int on, int off));
		MOCK_METHOD(void, set_motor_pwm, (int channel, int pwm));
};
