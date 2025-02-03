#include <QSignalSpy>
#include "SystemDataManager.hpp"
#include <gtest/gtest.h>

class SystemDataManagerTest : public ::testing::Test
{
protected:
    SystemDataManager *systemDataManager;

    void SetUp() override { systemDataManager = new SystemDataManager(); }

    void TearDown() override { delete systemDataManager; }
};

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

TEST_F(SystemDataManagerTest, WifiDataEmitsSignalOnChange)
{
    QSignalSpy wifiSpy(systemDataManager, &SystemDataManager::systemWifiUpdated);

    QString expectedStatus = "Connected";
    QString expectedName = "MyWiFi";

    systemDataManager->handleWifiData(expectedStatus, expectedName);

    ASSERT_EQ(wifiSpy.count(), 1);
    QList<QVariant> args = wifiSpy.takeFirst();
    ASSERT_EQ(args.at(0).toString(), expectedStatus);
    ASSERT_EQ(args.at(1).toString(), expectedName);

    // Sending the same data should NOT emit the signal again
    systemDataManager->handleWifiData(expectedStatus, expectedName);
    ASSERT_EQ(wifiSpy.count(), 0);
}

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

TEST_F(SystemDataManagerTest, IpAddressEmitsSignalOnChange)
{
    QSignalSpy ipSpy(systemDataManager, &SystemDataManager::ipAddressUpdated);

    QString expectedIp = "192.168.1.100";

    systemDataManager->handleIpAddressData(expectedIp);

    ASSERT_EQ(ipSpy.count(), 1);
    QList<QVariant> args = ipSpy.takeFirst();
    ASSERT_EQ(args.at(0).toString(), expectedIp);

    // Sending the same data should NOT emit the signal again
    systemDataManager->handleIpAddressData(expectedIp);
    ASSERT_EQ(ipSpy.count(), 0);
}

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
