/*!
 * @file II2CController.hpp
 * @brief Definition of the II2CController interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the II2CController interface, which
 * is responsible for controlling I2C devices.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef II2CCONTROLLER_HPP
#define II2CCONTROLLER_HPP

#include <cstdint>

/*!
 * @brief Interface for controlling I2C devices.
 * @class II2CController
 */
class II2CController
{
public:
    virtual ~II2CController() = default;
    virtual void writeRegister(uint8_t reg, uint16_t value) = 0;
    virtual uint16_t readRegister(uint8_t reg) = 0;
};

#endif // II2CCONTROLLER_HPP
