/*!
 * @file MockBatteryController.hpp
 * @brief File containing the Mock class of the BatteryController module.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the BatteryController module.
 * It uses Google Mock to create mock methods for the BatteryController module.
 * 
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef MOCKBATTERYCONTROLLER_HPP
#define MOCKBATTERYCONTROLLER_HPP

#include "IBatteryController.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockBatteryController
 * @brief Class to emulate the behavior of the BatteryController module.
 */
class MockBatteryController : public IBatteryController
{
public:
	/*! @brief Mocked method to get the battery percentage. */
	MOCK_METHOD(float, getBatteryPercentage, (), (override));
};

#endif // MOCKBATTERYCONTROLLER_HPP
