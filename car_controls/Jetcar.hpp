#include <iostream>
#include <thread>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <SDL2/SDL.h>

class Jetcar {

    private:
        // Constants
        const int MAX_ANGLE = 180;
        const int SERVO_CENTER_PWM = 345;
        const int SERVO_LEFT_PWM = 345 - 120;
        const int SERVO_RIGHT_PWM = 345 + 120;
        const int STEERING_CHANNEL = 0;

        // I2C variables
        int servo_bus_fd_;
        int motor_bus_fd_;
        int servo_addr_;
        int motor_addr_;

        // State
        std::atomic<bool> running_;
        std::atomic<int> current_speed_;
        std::atomic<int> current_angle_;

        // Joystick thread
        std::thread joystick_thread_;

    public:
        Jetcar();
        Jetcar(int servo_addr, int motor_addr);
        ~Jetcar();

        void start();
        void stop();
        void set_speed(int speed);
        void set_steering(int angle);
        void init_servo();
        void init_motors();
        void set_servo_pwm(int channel, int on_value, int off_value);
        void set_motor_pwm(int channel, int value);
        virtual void write_byte_data(int fd, int reg, int value);
        virtual int read_byte_data(int fd, int reg);
        void process_joystick();

        // Getters
        bool get_is_running() const { return running_; }
        int get_speed() const { return current_speed_; }
        int get_angle() const { return current_angle_; }
        int get_servo_bus_fd() const { return servo_bus_fd_; }
        int get_motor_bus_fd() const { return motor_bus_fd_; }
        int get_servo_addr() const { return servo_addr_; }
        int get_motor_addr() const { return motor_addr_; }
        int get_servo_center_pwm() const { return SERVO_CENTER_PWM; }
        int get_servo_left_pwm() const { return SERVO_LEFT_PWM; }
        int get_servo_right_pwm() const { return SERVO_RIGHT_PWM; }
        int get_steering_channel() const { return STEERING_CHANNEL; }
        int get_max_angle() const { return MAX_ANGLE; }
        int get_current_speed() const { return current_speed_; }
        int get_current_angle() const { return current_angle_; }
};
