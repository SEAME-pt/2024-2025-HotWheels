#ifndef BATTERYCONTROLLER_HPP
#define BATTERYCONTROLLER_HPP

#include <QObject>
#include "I2CController.hpp"
#include "IBatteryController.hpp"

class BatteryController : public QObject, public IBatteryController, private I2CController
{
    Q_OBJECT

public:
    explicit BatteryController(const char *i2c_device, int address, QObject *parent = nullptr);
    ~BatteryController() override = default;

    float getBatteryPercentage() override;

private:
    void setCalibration32V2A();
    float getBusVoltage_V();
    float getShuntVoltage_V();
};

#endif // BATTERYCONTROLLER_HPP
