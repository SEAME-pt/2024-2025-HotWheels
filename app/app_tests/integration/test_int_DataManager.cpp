/**
 * @file test_int_DataManager.cpp
 * @brief Integration tests for the DataManager class.
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
#include <QDebug>
#include <QSignalSpy>
#include "DataManager.hpp"
#include <gtest/gtest.h>

/**
 * @brief Class to test the integration between the DataManager and the
 * SystemDataManager, VehicleDataManager, and ClusterSettingsManager.
 * @class DataManagerTest
 */
class DataManagerTest : public ::testing::Test
{
protected:
	static QCoreApplication *app;
	DataManager *dataManager;

	static void SetUpTestSuite()
	{
		int argc = 0;
		char *argv[] = {nullptr};
		app = new QCoreApplication(argc, argv);
	}

	static void TearDownTestSuite() { delete app; }

	void SetUp() override
	{
		dataManager = new DataManager();
		ASSERT_NE(dataManager, nullptr);
		ASSERT_NE(dataManager->getSystemDataManager(), nullptr);
		ASSERT_NE(dataManager->getVehicleDataManager(), nullptr);
		ASSERT_NE(dataManager->getClusterSettingsManager(), nullptr);
	}

	void TearDown() override { delete dataManager; }
};

/** @brief Initialize static member */
QCoreApplication *DataManagerTest::app = nullptr;

/**
 * @test 🚗 Forward Speed Data
 * @brief Ensures that the DataManager forwards speed data to the VehicleDataManager.
 * @details This test verifies that the DataManager forwards speed data to the VehicleDataManager
 * by emitting the canDataProcessed signal.
 * @see DataManager::canDataProcessed
 */
TEST_F(DataManagerTest, ForwardSpeedDataToVehicleDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::canDataProcessed);

	dataManager->handleSpeedData(42.5f);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FLOAT_EQ(args.at(0).toFloat(), 42.5f);
}

/**
 * @test 🔄 Forward RPM Data
 * @brief Ensures that the DataManager forwards RPM data to the VehicleDataManager.
 * @details This test verifies that the DataManager forwards RPM data to the VehicleDataManager
 * by emitting the canDataProcessed signal.
 * @see DataManager::canDataProcessed
 */
TEST_F(DataManagerTest, ForwardRpmDataToVehicleDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::canDataProcessed);

	dataManager->handleRpmData(3500);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(1).toInt(), 3500);
}

/**
 * @test 🏎️ Forward Steering Data
 * @brief Ensures that the DataManager forwards steering data to the VehicleDataManager.
 * @details This test verifies that the DataManager forwards steering data to the VehicleDataManager
 * by emitting the engineDataProcessed signal.
 * @see DataManager::engineDataProcessed
 */
TEST_F(DataManagerTest, ForwardSteeringDataToVehicleDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::engineDataProcessed);

	dataManager->handleSteeringData(15);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(1).toInt(), 15);
}

/**
 * @test 🚦 Forward Direction Data
 * @brief Ensures that the DataManager forwards direction data to the VehicleDataManager.
 * @details This test verifies that the DataManager forwards direction data to the VehicleDataManager
 * by emitting the engineDataProcessed signal.
 * @see DataManager::engineDataProcessed
 */
TEST_F(DataManagerTest, ForwardDirectionDataToVehicleDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::engineDataProcessed);

	dataManager->handleDirectionData(CarDirection::Drive);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).value<CarDirection>(), CarDirection::Drive);
}

/**
 * @test 📅 Forward Time Data
 * @brief Ensures that the DataManager forwards time data to the SystemDataManager.
 * @details This test verifies that the DataManager forwards time data to the SystemDataManager
 * by emitting the systemTimeUpdated signal.
 * @see DataManager::systemTimeUpdated
 */
TEST_F(DataManagerTest, ForwardTimeDataToSystemDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::systemTimeUpdated);

	dataManager->handleTimeData("2025-01-31", "12:30", "Friday");
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).toString(), "2025-01-31");
	EXPECT_EQ(args.at(1).toString(), "12:30");
	EXPECT_EQ(args.at(2).toString(), "Friday");
}

/**
 * @test 📡 Forward WiFi Data
 * @brief Ensures that the DataManager forwards WiFi data to the SystemDataManager.
 * @details This test verifies that the DataManager forwards WiFi data to the SystemDataManager
 * by emitting the systemWifiUpdated signal.
 * @see DataManager::systemWifiUpdated
 */
TEST_F(DataManagerTest, ForwardWifiDataToSystemDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::systemWifiUpdated);

	dataManager->handleWifiData("Connected", "MyWiFi");
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).toString(), "Connected");
	EXPECT_EQ(args.at(1).toString(), "MyWiFi");
}

/**
 * @test 🌡 Forward Temperature Data
 * @brief Ensures that the DataManager forwards temperature data to the SystemDataManager.
 * @details This test verifies that the DataManager forwards temperature data to the SystemDataManager
 * by emitting the systemTemperatureUpdated signal.
 * @see DataManager::systemTemperatureUpdated
 */
TEST_F(DataManagerTest, ForwardTemperatureDataToSystemDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::systemTemperatureUpdated);

	dataManager->handleTemperatureData("25.5°C");
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).toString(), "25.5°C");
}

/**
 * @test 🌍 Forward IP Address Data
 * @brief Ensures that the DataManager forwards IP address data to the SystemDataManager.
 * @details This test verifies that the DataManager forwards IP address data to the SystemDataManager
 * by emitting the ipAddressUpdated signal.
 * @see DataManager::ipAddressUpdated
 */
TEST_F(DataManagerTest, ForwardIpAddressDataToSystemDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::ipAddressUpdated);

	dataManager->handleIpAddressData("192.168.1.100");
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).toString(), "192.168.1.100");
}

/**
 * @test 🔋 Forward Battery Percentage
 * @brief Ensures that the DataManager forwards battery percentage data to the SystemDataManager.
 * @details This test verifies that the DataManager forwards battery percentage data to the SystemDataManager
 * by emitting the batteryPercentageUpdated signal.
 * @see DataManager::batteryPercentageUpdated
 */
TEST_F(DataManagerTest, ForwardBatteryPercentageToSystemDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::batteryPercentageUpdated);

	dataManager->handleBatteryPercentage(87.5f);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FLOAT_EQ(args.at(0).toFloat(), 87.5f);
}

/**
 * @test 🚘 Forward Mileage Update
 * @brief Ensures that the DataManager forwards mileage data to the VehicleDataManager.
 * @details This test verifies that the DataManager forwards mileage data to the VehicleDataManager
 * by emitting the mileageUpdated signal.
 * @see DataManager::mileageUpdated
 */
TEST_F(DataManagerTest, ForwardMileageUpdateToVehicleDataManager)
{
	QSignalSpy spy(dataManager, &DataManager::mileageUpdated);

	dataManager->handleMileageUpdate(12345.67);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 12345.67);
}

/**
 * @test 🎛 Toggle Driving Mode
 * @brief Ensures that the DataManager toggles the driving mode.
 * @details This test verifies that the DataManager toggles the driving mode
 * by emitting the drivingModeUpdated signal.
 * @see DataManager::drivingModeUpdated
 */
TEST_F(DataManagerTest, ToggleDrivingMode)
{
	QSignalSpy spy(dataManager, &DataManager::drivingModeUpdated);

	dataManager->toggleDrivingMode();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}

/**
 * @test 🎨 Toggle Cluster Theme
 * @brief Ensures that the DataManager toggles the cluster theme.
 * @details This test verifies that the DataManager toggles the cluster theme
 * by emitting the clusterThemeUpdated signal.
 * @see DataManager::clusterThemeUpdated
 */
TEST_F(DataManagerTest, ToggleClusterTheme)
{
	QSignalSpy spy(dataManager, &DataManager::clusterThemeUpdated);

	dataManager->toggleClusterTheme();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}

/**
 * @test 📊 Toggle Cluster Metrics
 * @brief Ensures that the DataManager toggles the cluster metrics.
 * @details This test verifies that the DataManager toggles the cluster metrics
 * by emitting the clusterMetricsUpdated signal.
 * @see DataManager::clusterMetricsUpdated
 */
TEST_F(DataManagerTest, ToggleClusterMetrics)
{
	QSignalSpy spy(dataManager, &DataManager::clusterMetricsUpdated);

	dataManager->toggleClusterMetrics();
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}
