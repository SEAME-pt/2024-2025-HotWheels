#ifndef BATTERYCONTROLLER_HPP
#define BATTERYCONTROLLER_HPP

#include <QObject>
#include "I2CController.hpp"

class BatteryController : public QObject, private I2CController
{
    Q_OBJECT

public:
    explicit BatteryController(const char *i2c_device, int address, QObject *parent = nullptr);
    ~BatteryController() override = default;

    Q_INVOKABLE float getBatteryPercentage();

private:
    void setCalibration32V2A();
    float getBusVoltage_V();
    float getShuntVoltage_V();
};

#endif // BATTERYCONTROLLER_HPP
