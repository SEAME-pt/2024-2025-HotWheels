#include "BatteryController.hpp"
#include "I2CController.hpp"

// INA219 Register Addresses
#define REG_CALIBRATION 0x05
#define REG_BUSVOLTAGE 0x02
#define REG_SHUNTVOLTAGE 0x01

BatteryController::BatteryController(II2CController *i2cController)
    : m_i2cController(i2cController ? i2cController : new I2CController("/dev/i2c-1", 0x41))
    , m_ownI2CController(i2cController == nullptr)
{
    setCalibration32V2A();
}

BatteryController::~BatteryController()
{
    if (m_ownI2CController) {
        delete m_i2cController;
    }
}

void BatteryController::setCalibration32V2A()
{
    m_i2cController->writeRegister(REG_CALIBRATION, 4096);
}

float BatteryController::getBusVoltage_V()
{
    uint16_t raw = m_i2cController->readRegister(REG_BUSVOLTAGE);
    return ((raw >> 3) * 0.004); // Convert to volts
}

float BatteryController::getShuntVoltage_V()
{
    int16_t raw = static_cast<int16_t>(m_i2cController->readRegister(REG_SHUNTVOLTAGE));
    return raw * 0.01; // Convert to volts
}

float BatteryController::getBatteryPercentage()
{
    float busVoltage = getBusVoltage_V();
    float shuntVoltage = getShuntVoltage_V();
    float loadVoltage = busVoltage + shuntVoltage;

    // Calculate percentage
    float percentage = (loadVoltage - 6.0f) / 2.4f * 100.0f;
    if (percentage > 100.0f)
        percentage = 100.0f;
    if (percentage < 0.0f)
        percentage = 0.0f;
    return percentage;
}
