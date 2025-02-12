/*!
 * @file test_MileageManager.cpp
 * @brief Unit tests for the MileageManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the MileageManager class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QSignalSpy>
#include "MileageManager.hpp"
#include "MockMileageCalculator.hpp"
#include "MockMileageFileHandler.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

/*!
 * @class MileageManagerTest
 * @brief Test fixture for testing the MileageManager class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class MileageManagerTest : public ::testing::Test
{
protected:
	NiceMock<MockMileageCalculator> mockCalculator;
	NiceMock<MockMileageFileHandler> mockFileHandler;
	QString testFilePath = "test_mileage.txt";
	MileageManager *mileageManager;

	void SetUp() override
	{
		mileageManager = new MileageManager(testFilePath, &mockCalculator, &mockFileHandler);
	}

	void TearDown() override { delete mileageManager; }
};

/*!
 * @test Tests if the mileage manager initializes correctly.
 * @brief Ensures that the mileage manager initializes correctly.
 * @details This test verifies that the mileage manager initializes correctly.
 * 
 * @see MileageManager::initialize
*/
TEST_F(MileageManagerTest, Initialize_LoadsMileageFromFile)
{
	EXPECT_CALL(mockFileHandler, readMileage()).WillOnce(Return(123.45)); // Simulate stored mileage

	// Allow writeMileage to be called more than once due to shutdown
	EXPECT_CALL(mockFileHandler, writeMileage(123.45)).WillRepeatedly(Return());

	mileageManager->initialize();
	mileageManager->saveMileage();
}

/*!
 * @test Test OnSpeedUpdated method.
 * @brief Ensures that the mileage manager calls the calculator when the speed is updated.
 * @details This test verifies that the mileage manager calls the calculator when the speed is updated.
 * 
 * @see MileageManager::onSpeedUpdated
 */
TEST_F(MileageManagerTest, OnSpeedUpdated_CallsCalculator)
{
	EXPECT_CALL(mockCalculator, addSpeed(50.0)).Times(1);
	mileageManager->onSpeedUpdated(50.0);
}

/*!
 * @test Test UpdateMileage method.
 * @brief Ensures that the mileage manager updates the mileage correctly.
 * @details This test verifies that the mileage manager updates the mileage correctly.
 * 
 * @see MileageManager::updateMileage
 */
TEST_F(MileageManagerTest, UpdateMileage_EmitsMileageUpdatedSignal)
{
	EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(10.5));

	QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);
	ASSERT_TRUE(spy.isValid());

	mileageManager->updateMileage();

	ASSERT_EQ(spy.count(), 1);
	QList<QVariant> arguments = spy.takeFirst();
	EXPECT_DOUBLE_EQ(arguments.at(0).toDouble(), 10.5);
}

/*!
 * @test Test SaveMileage method.
 * @brief Ensures that the mileage manager saves the mileage correctly.
 * @details This test verifies that the mileage manager saves the mileage correctly.
 * 
 * @see MileageManager::saveMileage
 */
TEST_F(MileageManagerTest, SaveMileage_CallsFileHandler)
{
	EXPECT_CALL(mockFileHandler, writeMileage(200.0)).WillRepeatedly(Return());

	EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(50.0));
	mileageManager->updateMileage(); // Adds 50.0

	EXPECT_CALL(mockCalculator, calculateDistance()).WillOnce(Return(150.0));
	mileageManager->updateMileage(); // Adds another 150.0

	mileageManager->saveMileage();
}
