/*!
 * @file MockSystemInfoProvider.hpp
 * @brief File containing Mock classes to test the SystemInfoProvider module.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the SystemInfoProvider module.
 * It uses Google Mock to create mock methods for the SystemInfoProvider module.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKSYSTEMINFOPROVIDER_HPP
#define MOCKSYSTEMINFOPROVIDER_HPP

#include "ISystemInfoProvider.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockSystemInfoProvider
 * @brief Class to emulate the behavior of the SystemInfoProvider module.
 */
class MockSystemInfoProvider : public ISystemInfoProvider
{
public:
	/*! @brief Mocked method to get the battery percentage. */
	MOCK_METHOD(QString, getWifiStatus, (QString & wifiName), (const, override));
	/*! @brief Mocked method to get the battery percentage. */
	MOCK_METHOD(QString, getTemperature, (), (const, override));
	/*! @brief Mocked method to get the battery percentage. */
	MOCK_METHOD(QString, getIpAddress, (), (const, override));
};

#endif // MOCKSYSTEMINFOPROVIDER_HPP
