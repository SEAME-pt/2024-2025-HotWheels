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

#include <QObject>
#include "IBatteryController.hpp"
#include "II2CController.hpp"

/*!
 * @brief Class that manages the battery of the vehicle.
 * @class BatteryController
 */
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
