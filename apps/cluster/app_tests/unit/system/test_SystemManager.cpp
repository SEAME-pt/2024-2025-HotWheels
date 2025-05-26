/*!
 * @file test_SystemManager.cpp
 * @brief Unit tests for the SystemManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the SystemManager class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QSignalSpy>
#include "MockBatteryController.hpp"
#include "MockSystemInfoProvider.hpp"
#include "SystemManager.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;

/*!
 * @class SystemManagerTest
 * @brief Test fixture for testing the SystemManager class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class SystemManagerTest : public ::testing::Test
{
protected:
	NiceMock<MockSystemInfoProvider> mockInfoProvider;
	NiceMock<MockBatteryController> mockBatteryController;
	SystemManager *systemManager;

	void SetUp() override
	{
		systemManager = new SystemManager(&mockBatteryController, &mockInfoProvider, nullptr);
	}

	void TearDown() override { delete systemManager; }
};

/*!
 * @test Tests if the time is correctly updated.
 * @brief Ensures that the time is correctly updated.
 * @details This test verifies that the time is correctly updated.
 *
 * @see SystemManager::updateTime
 */
TEST_F(SystemManagerTest, UpdateTime_EmitsCorrectSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::timeUpdated);
	ASSERT_TRUE(spy.isValid());

	systemManager->updateTime();

	ASSERT_EQ(spy.count(), 1);
	QList<QVariant> arguments = spy.takeFirst();
	EXPECT_FALSE(arguments.at(0).toString().isEmpty()); // Date
	EXPECT_FALSE(arguments.at(1).toString().isEmpty()); // Time
	EXPECT_FALSE(arguments.at(2).toString().isEmpty()); // Day
}

/*!
 * @test Tests if the system status is correctly updated.
 * @brief Ensures that the system status is correctly updated.
 * @details This test verifies that the system status is correctly updated.
 *
 * @see SystemManager::updateSystemStatus
 */
TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsWifiStatus)
{
	QString mockWifiName = "MyWiFi";

	EXPECT_CALL(mockInfoProvider, getWifiStatus(_))
		.WillOnce(DoAll(testing::SetArgReferee<0>(mockWifiName), // Set wifiName argument
						Return(QString("Connected"))             // Return "Connected"
						));

	QSignalSpy spy(systemManager, &SystemManager::wifiStatusUpdated);
	ASSERT_TRUE(spy.isValid());

	systemManager->updateSystemStatus();

	ASSERT_EQ(spy.count(), 1);
	QList<QVariant> arguments = spy.takeFirst();
	EXPECT_EQ(arguments.at(0).toString(), "Connected");
	EXPECT_EQ(arguments.at(1).toString(), "MyWiFi");
}

/*!
 * @test Tests if the system status is correctly updated.
 * @brief Ensures that the system status is correctly updated.
 * @details This test verifies that the system status is correctly updated.
 *
 * @see SystemManager::updateSystemStatus
 */
TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsTemperature)
{
	EXPECT_CALL(mockInfoProvider, getTemperature()).WillOnce(Return(QString("42.0°C")));

	QSignalSpy spy(systemManager, &SystemManager::temperatureUpdated);
	ASSERT_TRUE(spy.isValid());

	systemManager->updateSystemStatus();

	ASSERT_EQ(spy.count(), 1);
	EXPECT_EQ(spy.takeFirst().at(0).toString(), "42.0°C");
}

/*!
 * @test Tests if the system status is correctly updated.
 * @brief Ensures that the system status is correctly updated.
 * @details This test verifies that the system status is correctly updated.
 *
 * @see SystemManager::updateSystemStatus
 */
TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsBatteryPercentage)
{
	EXPECT_CALL(mockBatteryController, getBatteryPercentage()).WillOnce(Return(75.0f));

	QSignalSpy spy(systemManager, &SystemManager::batteryPercentageUpdated);
	ASSERT_TRUE(spy.isValid());

	systemManager->updateSystemStatus();

	ASSERT_EQ(spy.count(), 1);
	EXPECT_FLOAT_EQ(spy.takeFirst().at(0).toFloat(), 75.0f);
}
