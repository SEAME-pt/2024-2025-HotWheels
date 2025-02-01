/**
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

/** @def REG_CALIBRATION The register address for the calibration register. */
#define REG_CALIBRATION 0x05
/** @def REG_BUSVOLTAGE The register address for the bus voltage register. */
#define REG_BUSVOLTAGE 0x02
/** @def REG_SHUNTVOLTAGE The register address for the shunt voltage register.
 */
#define REG_SHUNTVOLTAGE 0x01

/**
 * @brief Construct a new BatteryController object.
 * @param i2c_device The I2C device to use for communication.
 * @param address The I2C address of the INA219.
 * @param parent The parent QObject.
 * @details This constructor initializes the BatteryController object with the
 * specified I2C device and address.
 */
BatteryController::BatteryController(const char *i2c_device, int address,
                                     QObject *parent)
    : QObject(parent), I2CController(i2c_device, address) {
  setCalibration32V2A();
}

/**
 * @brief Set the calibration for the INA219.
 * @details This function sets the calibration for the INA219 to 32V and 2A.
 */
void BatteryController::setCalibration32V2A() {
  writeRegister(REG_CALIBRATION, 4096);
}

/**
 * @brief Read a 16-bit register from the INA219.
 * @param reg The register address to read.
 * @return uint16_t The value read from the register.
 * @details This function reads a 16-bit register from the INA219.
 */
float BatteryController::getBusVoltage_V() {
  uint16_t raw = readRegister(REG_BUSVOLTAGE);
  return ((raw >> 3) * 0.004); // Convert to volts
}

/**
 * @brief Read a 16-bit register from the INA219.
 * @param reg The register address to read.
 * @return uint16_t The value read from the register.
 * @details This function reads a 16-bit register from the INA219.
 */
float BatteryController::getShuntVoltage_V() {
  int16_t raw = static_cast<int16_t>(readRegister(REG_SHUNTVOLTAGE));
  return raw * 0.01; // Convert to volts
}

/**
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
