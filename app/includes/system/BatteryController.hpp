#ifndef BATTERYCONTROLLER_HPP
#define BATTERYCONTROLLER_HPP

#include <QObject>
#include "IBatteryController.hpp"
#include "II2CController.hpp"

class BatteryController : public IBatteryController
{
public:
    explicit BatteryController(II2CController *i2cController = nullptr);
    ~BatteryController() override;

    float getBatteryPercentage() override;

private:
    void setCalibration32V2A();
    float getBusVoltage_V();
    float getShuntVoltage_V();

    II2CController *m_i2cController;
    bool m_ownI2CController;
};

#endif // BATTERYCONTROLLER_HPP
