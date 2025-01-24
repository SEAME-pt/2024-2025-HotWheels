#ifndef PERIPHERALCONTROLLER_HPP
#define PERIPHERALCONTROLLER_HPP

#include <QObject>
#include "enums.hpp"
#include <atomic>
#include <unistd.h>
#include <QDebug>
#include <atomic>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

class PeripheralController : public QObject
{
    Q_OBJECT

    public:
        PeripheralController(QObject *parent = nullptr);
        PeripheralController(int servo_addr, int motor_addr, QObject *parent = nullptr);
        ~PeripheralController();

        int i2c_smbus_write_byte_data(int file, uint8_t command, uint8_t value);
        int i2c_smbus_read_byte_data(int file, uint8_t command);

        virtual void write_byte_data(int fd, int reg, int value, bool disabled);
        virtual int read_byte_data(int fd, int reg, bool disabled);

        void set_servo_pwm(int channel, int on_value, int off_value, int servo_bus_fd_, bool disabled);
        void set_motor_pwm(int channel, int value, int motor_bus_fd_, bool disabled);

        void init_servo(int servo_bus_fd_, bool disabled);
        void init_motors(int motor_bus_fd_, bool disabled);
};

#endif
