#ifndef IPERIPHERALCONTROLLER_HPP
#define IPERIPHERALCONTROLLER_HPP

#include <QObject>
#include <unistd.h>
#include <QDebug>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

class IPeripheralController
{
    public:
        virtual ~IPeripheralController() = default;

        virtual int i2c_smbus_write_byte_data(int file, uint8_t command, uint8_t value) = 0;
        virtual int i2c_smbus_read_byte_data(int file, uint8_t command) = 0;

        virtual void write_byte_data(int fd, int reg, int value) = 0;
        virtual int read_byte_data(int fd, int reg) = 0;

        virtual void set_servo_pwm(int channel, int on_value, int off_value) = 0;
        virtual void set_motor_pwm(int channel, int value) = 0;

        virtual void init_servo() = 0;
        virtual void init_motors() = 0;
};

#endif
