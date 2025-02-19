/*!
 * @file MockMileageCalculator.hpp
 * @brief File containing the Mock class of the MileageCalculator class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the MileageCalculator class.
 * It uses Google Mock to create mock methods for the MileageCalculator class.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKMILEAGECALCULATOR_HPP
#define MOCKMILEAGECALCULATOR_HPP

#include "IMileageCalculator.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockMileageCalculator
 * @brief Class to emulate the behavior of the MileageCalculator class.
 */
class MockMileageCalculator : public IMileageCalculator
{
public:
	/*! @brief Mocked method to add a speed to the MileageCalculator. */
	MOCK_METHOD(void, addSpeed, (float speed), (override));
	/*! @brief Mocked method to add a time to the MileageCalculator. */
	MOCK_METHOD(double, calculateDistance, (), (override));
};

#endif // MOCKMILEAGECALCULATOR_HPP
