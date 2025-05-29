/*!
 * @file test_SystemDataManager.cpp
 * @brief Unit tests for the SystemDataManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the SystemDataManager class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QSignalSpy>
#include "SystemDataManager.hpp"
#include <gtest/gtest.h>

/*!
 * @class SystemDataManagerTest
 * @brief Test fixture for testing the SystemDataManager class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class SystemDataManagerTest : public ::testing::Test
{
protected:
	SystemDataManager *systemDataManager;

	void SetUp() override { systemDataManager = new SystemDataManager(); }

	void TearDown() override { delete systemDataManager; }
};

/*!
 * @test Tests if the time data emits a signal.
 * @brief Ensures that the time data emits a signal when received.
 *
 * @details This test verifies that the time data emits a signal when received.
 *
 * @see SystemDataManager::handleTimeData
 */
TEST_F(SystemDataManagerTest, TimeDataEmitsSignal)
{
	QSignalSpy timeSpy(systemDataManager, &SystemDataManager::systemTimeUpdated);

	QString expectedDate = "2025-01-30";
	QString expectedTime = "14:30:00";
	QString expectedDay = "Thursday";

	systemDataManager->handleTimeData(expectedDate, expectedTime, expectedDay);

	ASSERT_EQ(timeSpy.count(), 1);
	QList<QVariant> args = timeSpy.takeFirst();
	ASSERT_EQ(args.at(0).toString(), expectedDate);
	ASSERT_EQ(args.at(1).toString(), expectedTime);
	ASSERT_EQ(args.at(2).toString(), expectedDay);
}

/*!
 * @test Tests if the temperature data emits a signal when changed.
 * @brief Ensures that the temperature data emits a signal when changed.
 *
 * @details This test verifies that the temperature data emits a signal when changed.
 *
 * @see SystemDataManager::handleTemperatureData
 */
TEST_F(SystemDataManagerTest, TemperatureDataEmitsSignalOnChange)
{
	QSignalSpy tempSpy(systemDataManager, &SystemDataManager::systemTemperatureUpdated);

	QString expectedTemp = "25.5°C";

	systemDataManager->handleTemperatureData(expectedTemp);

	ASSERT_EQ(tempSpy.count(), 1);
	QList<QVariant> args = tempSpy.takeFirst();
	ASSERT_EQ(args.at(0).toString(), expectedTemp);

	// Sending the same data should NOT emit the signal again
	systemDataManager->handleTemperatureData(expectedTemp);
	ASSERT_EQ(tempSpy.count(), 0);
}

/*!
 * @test Tests if the battery percentage emits a signal when changed.
 * @brief Ensures that the battery percentage emits a signal when changed.
 *
 * @details This test verifies that the battery percentage emits a signal when changed.
 *
 * @see SystemDataManager::handleBatteryPercentage
 */
TEST_F(SystemDataManagerTest, BatteryPercentageEmitsSignalOnChange)
{
	QSignalSpy batterySpy(systemDataManager, &SystemDataManager::batteryPercentageUpdated);

	float expectedBattery = 78.5f;

	systemDataManager->handleBatteryPercentage(expectedBattery);

	ASSERT_EQ(batterySpy.count(), 1);
	QList<QVariant> args = batterySpy.takeFirst();
	ASSERT_FLOAT_EQ(args.at(0).toFloat(), expectedBattery);

	// Sending the same data should NOT emit the signal again
	systemDataManager->handleBatteryPercentage(expectedBattery);
	ASSERT_EQ(batterySpy.count(), 0);
}
