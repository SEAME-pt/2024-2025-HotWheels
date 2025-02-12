/*!
 * @file MockI2CController.hpp
 * @brief File containing Mock classes to test the controller of the I2C module.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the I2CController module.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKI2CCONTROLLER_HPP
#define MOCKI2CCONTROLLER_HPP

#include "II2CController.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockI2CController
 * @brief Class to emulate the behavior of the I2C controller.
 */
class MockI2CController : public II2CController
{
public:
	/*! @brief Mocked method to initialize the I2C controller. */
	MOCK_METHOD(void, writeRegister, (uint8_t reg, uint16_t value), (override));
	/*! @brief Mocked method to read a register from the I2C controller. */
	MOCK_METHOD(uint16_t, readRegister, (uint8_t reg), (override));
};

#endif // MOCKI2CCONTROLLER_HPP
