/**
 * @file test_int_SystemManager.cpp
 * @brief Integration tests for the SystemManager class.
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
#include "BatteryController.hpp"
#include "SystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"
#include "SystemManager.hpp"
#include <gtest/gtest.h>

/**
 * @brief Class to test the integration between the SystemManager and the
 * BatteryController, SystemInfoProvider, and SystemCommandExecutor.
 * @class SystemManagerTest
 */
class SystemManagerTest : public ::testing::Test
{
protected:
	static QCoreApplication *app;
	SystemManager *systemManager;
	IBatteryController *batteryController;
	ISystemInfoProvider *systemInfoProvider;
	ISystemCommandExecutor *systemCommandExecutor;

	static void SetUpTestSuite()
	{
		int argc = 0;
		char *argv[] = {nullptr};
		app = new QCoreApplication(argc, argv);
	}

	static void TearDownTestSuite() { delete app; }

	void SetUp() override
	{
		batteryController = new BatteryController();
		systemInfoProvider = new SystemInfoProvider();
		systemCommandExecutor = new SystemCommandExecutor();
		systemManager = new SystemManager(batteryController,
										  systemInfoProvider,
										  systemCommandExecutor);
	}

	void TearDown() override
	{
		delete systemManager;
		delete batteryController;
		delete systemInfoProvider;
		delete systemCommandExecutor;
	}
};

/** @brief Initialize static member */
QCoreApplication *SystemManagerTest::app = nullptr;

/**
 * @test 🚀 Initialize System Manager
 * @brief Ensures that the SystemManager initializes successfully.
 * @details This test verifies that the SystemManager initializes successfully
 * by emitting the timeUpdated, wifiStatusUpdated, temperatureUpdated,
 * batteryPercentageUpdated, and ipAddressUpdated signals.
 * @see SystemManager::timeUpdated
 */
TEST_F(SystemManagerTest, UpdateTimeSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::timeUpdated);

	systemManager->initialize();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FALSE(args.isEmpty());
}

/**
 * @test 📶 Update Wifi Status Signal
 * @brief Ensures that the SystemManager updates the wifi status.
 * @details This test verifies that the SystemManager updates the wifi status
 * by emitting the wifiStatusUpdated signal.
 * @see SystemManager::wifiStatusUpdated
 */
TEST_F(SystemManagerTest, UpdateWifiStatusSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::wifiStatusUpdated);

	systemManager->initialize();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FALSE(args.isEmpty());
}

/**
 * @test 🌡 Update Temperature Signal
 * @brief Ensures that the SystemManager updates the temperature.
 * @details This test verifies that the SystemManager updates the temperature
 * by emitting the temperatureUpdated signal.
 * @see SystemManager::temperatureUpdated
 */
TEST_F(SystemManagerTest, UpdateTemperatureSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::temperatureUpdated);

	systemManager->initialize();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FALSE(args.isEmpty());
}

/**
 * @test 🔋 Update Battery Percentage Signal
 * @brief Ensures that the SystemManager updates the battery percentage.
 * @details This test verifies that the SystemManager updates the battery
 * percentage by emitting the batteryPercentageUpdated signal.
 * @see SystemManager::batteryPercentageUpdated
 */
TEST_F(SystemManagerTest, UpdateBatteryPercentageSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::batteryPercentageUpdated);

	systemManager->initialize();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_GE(args.at(0).toFloat(), 0.0f);
	EXPECT_LE(args.at(0).toFloat(), 100.0f);
}

/**
 * @test 🌍 Update IP Address Signal
 * @brief Ensures that the SystemManager updates the IP address.
 * @details This test verifies that the SystemManager updates the IP address
 * by emitting the ipAddressUpdated signal.
 * @see SystemManager::ipAddressUpdated
 */
TEST_F(SystemManagerTest, UpdateIpAddressSignal)
{
	QSignalSpy spy(systemManager, &SystemManager::ipAddressUpdated);

	systemManager->initialize();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0);
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FALSE(args.isEmpty());
}

/**
 * @test 🚀 Shutdown System Manager
 * @brief Ensures that the SystemManager shuts down successfully.
 * @details This test verifies that the SystemManager shuts down successfully
 * by deactivating the time timer and status timer.
 */
TEST_F(SystemManagerTest, ShutdownSystemManager)
{
	systemManager->initialize();
	systemManager->shutdown();

	EXPECT_EQ(systemManager->getTimeTimer().isActive(), false);
	EXPECT_EQ(systemManager->getStatusTimer().isActive(), false);
}
