/*!
 * @file BatteryController.hpp
 * @brief Definition of the BatteryController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the BatteryController class,
 * which is used to manage the battery of the vehicle.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef BATTERYCONTROLLER_HPP
#define BATTERYCONTROLLER_HPP

#include "I2CController.hpp"
#include <QObject>

/*!
 * @brief Class that manages the battery of a vehicle.
 * @class BatteryController inherits from QObject and I2CController
 */
class BatteryController : public QObject, private I2CController {
  Q_OBJECT

public:
  explicit BatteryController(const char *i2c_device, int address,
                             QObject *parent = nullptr);
  ~BatteryController() override = default;
  Q_INVOKABLE float getBatteryPercentage();

private:
  void setCalibration32V2A();
  float getBusVoltage_V();
  float getShuntVoltage_V();
};

#endif // BATTERYCONTROLLER_HPP
