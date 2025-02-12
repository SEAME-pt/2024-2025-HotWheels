/**
 * @file test_int_MileageManager.cpp
 * @brief Integration tests for the MileageManager class.
 * @version 0.1
 * @date 2025-02-12
 * @author Michel Batista (@MicchelFAB)
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <QCoreApplication>
#include <QSignalSpy>
#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"
#include "MileageManager.hpp"
#include <gtest/gtest.h>

/**
 * @brief Class to test the integration between the MileageManager and the
 * MileageCalculator and MileageFileHandler.
 * @class MileageManagerTest
 */
class MileageManagerTest : public ::testing::Test
{
protected:
	static QCoreApplication *app;
	MileageManager *mileageManager;
	IMileageCalculator *calculator;
	IMileageFileHandler *fileHandler;

	static void SetUpTestSuite()
	{
		int argc = 0;
		char *argv[] = {nullptr};
		app = new QCoreApplication(argc, argv);
	}

	static void TearDownTestSuite() { delete app; }

	void SetUp() override
	{
		calculator = new MileageCalculator();
		fileHandler = new MileageFileHandler("/home/hotweels/app_data/test_mileage.json");
		mileageManager = new MileageManager("/home/hotweels/app_data/test_mileage.json",
											calculator,
											fileHandler);
	}

	void TearDown() override
	{
		delete mileageManager;
		delete calculator;
		delete fileHandler;
	}
};

/** @brief Initialize static member */
QCoreApplication *MileageManagerTest::app = nullptr;

/** 
 * @test 🔄 Forward Mileage Data
 * @brief Ensures that the MileageManager forwards mileage data.
 * @details This test verifies that the MileageManager forwards mileage data
 * by emitting the mileageUpdated signal.
 * @see MileageManager::mileageUpdated
 */
TEST_F(MileageManagerTest, ForwardMileageData)
{
	QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

	mileageManager->onSpeedUpdated(10.0f);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 10.0);
}

/** 
 * @test 🚗 Initialize Mileage Manager
 * @brief Ensures that the MileageManager initializes successfully.
 * @details This test verifies that the MileageManager initializes successfully
 * by emitting the mileageUpdated signal.
 * @see MileageManager::mileageUpdated
 */
TEST_F(MileageManagerTest, InitializeMileageManager)
{
	mileageManager->initialize();
	QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

	QCoreApplication::processEvents();
	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 0.0);
}

/** 
 * @test 🏎️💨 Update Mileage on Speed Update
 * @brief Ensures that the MileageManager updates the mileage on speed update.
 * @details This test verifies that the MileageManager updates the mileage on
 * speed update by emitting the mileageUpdated signal.
 * @see MileageManager::mileageUpdated
 */
TEST_F(MileageManagerTest, UpdateMileageOnSpeedUpdate)
{
	QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

	mileageManager->onSpeedUpdated(10.0f);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 10.0);
}

/** 
 * @test 💾 Save Mileage
 * @brief Ensures that the MileageManager saves the mileage.
 * @details This test verifies that the MileageManager saves the mileage by
 * calling the saveMileage() method.
 * @see MileageManager::saveMileage
 */
TEST_F(MileageManagerTest, SaveMileage)
{
	mileageManager->onSpeedUpdated(5.0f);
	QCoreApplication::processEvents();

	mileageManager->saveMileage();
	QCoreApplication::processEvents();

	double savedMileage = mileageManager->getTotalMileage();
	EXPECT_DOUBLE_EQ(savedMileage, 5.0);
}

/** 
 * @test ⏱ Update Timer Interval
 * @brief Ensures that the MileageManager updates the timer interval.
 * @details This test verifies that the MileageManager updates the timer interval
 * by emitting the mileageUpdated signal.
 * @see MileageManager::mileageUpdated
 */
TEST_F(MileageManagerTest, UpdateTimerInterval)
{
	QSignalSpy spy(mileageManager, &MileageManager::mileageUpdated);

	mileageManager->initialize();
	QTimer::singleShot(1000, QCoreApplication::instance(), &QCoreApplication::quit);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
}

/** 
 * @test ⏻ Shutdown Mileage Manager
 * @brief Ensures that the MileageManager shuts down successfully.
 * @details This test verifies that the MileageManager shuts down successfully
 * by calling the shutdown() method.
 * @see MileageManager::shutdown
 */
TEST_F(MileageManagerTest, ShutdownMileageManager)
{
	mileageManager->initialize();
	mileageManager->shutdown();

	double finalMileage = mileageManager->getTotalMileage();
	EXPECT_DOUBLE_EQ(finalMileage, 0.0);
}
