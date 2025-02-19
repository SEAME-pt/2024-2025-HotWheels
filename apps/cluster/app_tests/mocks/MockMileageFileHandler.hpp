/*!
 * @file MockMileageFileHandler.hpp
 * @brief File containing the Mock class of the MileageFileHandler class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the MileageFileHandler class.
 * It uses Google Mock to create mock methods for the MileageFileHandler class.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKMILEAGEFILEHANDLER_HPP
#define MOCKMILEAGEFILEHANDLER_HPP

#include "IMileageFileHandler.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockMileageFileHandler
 * @brief Class to emulate the behavior of the MileageFileHandler class.
 */
class MockMileageFileHandler : public IMileageFileHandler
{
public:
	/*! @brief Mocked method to read the mileage from the file. */
	MOCK_METHOD(double, readMileage, (), (const, override));
	/*! @brief Mocked method to write the mileage to the file. */
	MOCK_METHOD(void, writeMileage, (double mileage), (const, override));
};

#endif // MOCKMILEAGEFILEHANDLER_HPP
