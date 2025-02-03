/*!
 * @file BatteryController.cpp
 * @brief Implementation of the BatteryController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the BatteryController
 * class, which is used to control the battery of the vehicle.
 * @note This class is used to control the battery of the vehicle, including
 * monitoring the battery level and charging status.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the INA219 is properly connected and configured.
 * @see BatteryController.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "BatteryController.hpp"
#include "I2CController.hpp"

/*! @def REG_CALIBRATION The register address for the calibration register. */
#define REG_CALIBRATION 0x05
/*! @def REG_BUSVOLTAGE The register address for the bus voltage register. */
#define REG_BUSVOLTAGE 0x02
/*! @def REG_SHUNTVOLTAGE The register address for the shunt voltage register.
 */
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

/*!
 * @brief Get the battery percentage.
 * @return float The battery percentage.
 * @details This function calculates the battery percentage based on the bus and
 * shunt voltages.
 */
float BatteryController::getBatteryPercentage() {
  float busVoltage = getBusVoltage_V();
  float shuntVoltage = getShuntVoltage_V();
  float loadVoltage = busVoltage + shuntVoltage;

  // Calculate percentage
  float percentage = (loadVoltage - 6.0F) / 2.4f * 100.0F;
  if (percentage > 100.0F)
    percentage = 100.0F;
  if (percentage < 0.0F)
    percentage = 0.0F;
  return percentage;
}
