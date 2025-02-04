#include <QCoreApplication>
#include <QDebug>
#include <QSignalSpy>
#include "DataManager.hpp"
#include <gtest/gtest.h>

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

// Initialize static member
QCoreApplication *DataManagerTest::app = nullptr;

// 🚗 **Test: Forward Speed Data**
TEST_F(DataManagerTest, ForwardSpeedDataToVehicleDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::canDataProcessed);

    dataManager->handleSpeedData(42.5f);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FLOAT_EQ(args.at(0).toFloat(), 42.5f);
}

// 🔄 **Test: Forward RPM Data**
TEST_F(DataManagerTest, ForwardRpmDataToVehicleDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::canDataProcessed);

    dataManager->handleRpmData(3500);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(1).toInt(), 3500);
}

// 🏎️ **Test: Forward Steering Data**
TEST_F(DataManagerTest, ForwardSteeringDataToVehicleDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::engineDataProcessed);

    dataManager->handleSteeringData(15);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(1).toInt(), 15);
}

// ➡️ **Test: Forward Direction Data**
TEST_F(DataManagerTest, ForwardDirectionDataToVehicleDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::engineDataProcessed);

    dataManager->handleDirectionData(CarDirection::Drive);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(0).value<CarDirection>(), CarDirection::Drive);
}

// 📅 **Test: Forward Time Data**
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

// 📡 **Test: Forward WiFi Data**
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

// 🌡 **Test: Forward Temperature Data**
TEST_F(DataManagerTest, ForwardTemperatureDataToSystemDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::systemTemperatureUpdated);

    dataManager->handleTemperatureData("25.5°C");
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(0).toString(), "25.5°C");
}

// 🌍 **Test: Forward IP Address Data**
TEST_F(DataManagerTest, ForwardIpAddressDataToSystemDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::ipAddressUpdated);

    dataManager->handleIpAddressData("192.168.1.100");
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_EQ(args.at(0).toString(), "192.168.1.100");
}

// 🔋 **Test: Forward Battery Percentage**
TEST_F(DataManagerTest, ForwardBatteryPercentageToSystemDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::batteryPercentageUpdated);

    dataManager->handleBatteryPercentage(87.5f);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_FLOAT_EQ(args.at(0).toFloat(), 87.5f);
}

// 🚘 **Test: Forward Mileage Update**
TEST_F(DataManagerTest, ForwardMileageUpdateToVehicleDataManager)
{
    QSignalSpy spy(dataManager, &DataManager::mileageUpdated);

    dataManager->handleMileageUpdate(12345.67);
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
    QList<QVariant> args = spy.takeFirst();
    EXPECT_DOUBLE_EQ(args.at(0).toDouble(), 12345.67);
}

// 🎛 **Test: Toggle Driving Mode**
TEST_F(DataManagerTest, ToggleDrivingMode)
{
    QSignalSpy spy(dataManager, &DataManager::drivingModeUpdated);

    dataManager->toggleDrivingMode();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}

// 🎨 **Test: Toggle Cluster Theme**
TEST_F(DataManagerTest, ToggleClusterTheme)
{
    QSignalSpy spy(dataManager, &DataManager::clusterThemeUpdated);

    dataManager->toggleClusterTheme();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}

// 📊 **Test: Toggle Cluster Metrics**
TEST_F(DataManagerTest, ToggleClusterMetrics)
{
    QSignalSpy spy(dataManager, &DataManager::clusterMetricsUpdated);

    dataManager->toggleClusterMetrics();
    QCoreApplication::processEvents();

    ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
}
