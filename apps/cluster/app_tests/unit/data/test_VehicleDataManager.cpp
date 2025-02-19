/*!
 * @file test_VehicleDataManager.cpp
 * @brief Unit tests for the VehicleDataManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the VehicleDataManager class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QSignalSpy>
#include "VehicleDataManager.hpp"
#include <gtest/gtest.h>

/*!
 * @class VehicleDataManagerTest
 * @brief Test fixture for testing the VehicleDataManager class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class VehicleDataManagerTest : public ::testing::Test
{
protected:
	VehicleDataManager *vehicleDataManager;

	void SetUp() override { vehicleDataManager = new VehicleDataManager(); }

	void TearDown() override { delete vehicleDataManager; }
};

/*!
 * @test Tests if the RPM data emits a signal when changed.
 * @brief Ensures that the RPM data emits a signal when changed.
 *
 * @details This test verifies that the RPM data emits a signal when changed.
 *
 * @see VehicleDataManager::handleRpmData
 */
TEST_F(VehicleDataManagerTest, RpmDataEmitsSignal)
{
	QSignalSpy canDataSpy(vehicleDataManager, &VehicleDataManager::canDataProcessed);

	int expectedRpm = 3000;

	vehicleDataManager->handleRpmData(expectedRpm);

	ASSERT_EQ(canDataSpy.count(), 1);
	QList<QVariant> args = canDataSpy.takeFirst();
	ASSERT_FLOAT_EQ(args.at(0).toFloat(), 0.0f); // Speed remains unchanged
	ASSERT_EQ(args.at(1).toInt(), expectedRpm);
}

/*!
 * @test Tests if the speed data emits a signal when changed.
 * @brief Ensures that the speed data emits a signal when changed.
 *
 * @details This test verifies that the speed data emits a signal when changed.
 *
 * @see VehicleDataManager::handleSpeedData
 */
TEST_F(VehicleDataManagerTest, SpeedDataEmitsSignalInKilometers)
{
	QSignalSpy canDataSpy(vehicleDataManager, &VehicleDataManager::canDataProcessed);

	float expectedSpeed = 120.5f;

	vehicleDataManager->handleSpeedData(expectedSpeed);

	ASSERT_EQ(canDataSpy.count(), 1);
	QList<QVariant> args = canDataSpy.takeFirst();
	ASSERT_FLOAT_EQ(args.at(0).toFloat(), expectedSpeed);
	ASSERT_EQ(args.at(1).toInt(), 0); // RPM remains unchanged
}

/*!
 * @test Tests if the mileage data emits a signal when changed.
 * @brief Ensures that the mileage data emits a signal when changed.
 *
 * @details This test verifies that the mileage data emits a signal when changed.
 *
 * @see VehicleDataManager::handleMileageUpdate
 */
TEST_F(VehicleDataManagerTest, MileageDataEmitsSignalOnChange)
{
	QSignalSpy mileageSpy(vehicleDataManager, &VehicleDataManager::mileageUpdated);

	double expectedMileage = 5000.75;

	vehicleDataManager->handleMileageUpdate(expectedMileage);

	ASSERT_EQ(mileageSpy.count(), 1);
	QList<QVariant> args = mileageSpy.takeFirst();
	ASSERT_DOUBLE_EQ(args.at(0).toDouble(), expectedMileage);

	// Sending the same data should NOT emit the signal again
	vehicleDataManager->handleMileageUpdate(expectedMileage);
	ASSERT_EQ(mileageSpy.count(), 0);
}

/*!
 * @test Tests if the direction data emits a signal when changed.
 * @brief Ensures that the direction data emits a signal when changed.
 *
 * @details This test verifies that the direction data emits a signal when changed.
 *
 * @see VehicleDataManager::handleDirectionData
 */
TEST_F(VehicleDataManagerTest, DirectionDataEmitsSignalOnChange)
{
	QSignalSpy engineDataSpy(vehicleDataManager, &VehicleDataManager::engineDataProcessed);

	CarDirection expectedDirection = CarDirection::Drive;

	vehicleDataManager->handleDirectionData(expectedDirection);

	ASSERT_EQ(engineDataSpy.count(), 1);
	QList<QVariant> args = engineDataSpy.takeFirst();
	ASSERT_EQ(args.at(0).value<CarDirection>(), expectedDirection);
	ASSERT_EQ(args.at(1).toInt(), 0); // Steering remains unchanged

	// Sending the same direction should NOT emit the signal again
	vehicleDataManager->handleDirectionData(expectedDirection);
	ASSERT_EQ(engineDataSpy.count(), 0);
}

/*!
 * @test Tests if the steering data emits a signal when changed.
 * @brief Ensures that the steering data emits a signal when changed.
 *
 * @details This test verifies that the steering data emits a signal when changed.
 *
 * @see VehicleDataManager::handleSteeringData
 */
TEST_F(VehicleDataManagerTest, SteeringDataEmitsSignalOnChange)
{
	QSignalSpy engineDataSpy(vehicleDataManager, &VehicleDataManager::engineDataProcessed);

	int expectedAngle = 15;

	vehicleDataManager->handleSteeringData(expectedAngle);

	ASSERT_EQ(engineDataSpy.count(), 1);
	QList<QVariant> args = engineDataSpy.takeFirst();
	ASSERT_EQ(args.at(0).value<CarDirection>(), CarDirection::Stop); // Direction remains unchanged
	ASSERT_EQ(args.at(1).toInt(), expectedAngle);

	// Sending the same steering angle should NOT emit the signal again
	vehicleDataManager->handleSteeringData(expectedAngle);
	ASSERT_EQ(engineDataSpy.count(), 0);
}
