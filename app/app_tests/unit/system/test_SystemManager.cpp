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

TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsTemperature)
{
    EXPECT_CALL(mockInfoProvider, getTemperature()).WillOnce(Return(QString("42.0°C")));

    QSignalSpy spy(systemManager, &SystemManager::temperatureUpdated);
    ASSERT_TRUE(spy.isValid());

    systemManager->updateSystemStatus();

    ASSERT_EQ(spy.count(), 1);
    EXPECT_EQ(spy.takeFirst().at(0).toString(), "42.0°C");
}

TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsBatteryPercentage)
{
    EXPECT_CALL(mockBatteryController, getBatteryPercentage()).WillOnce(Return(75.0f));

    QSignalSpy spy(systemManager, &SystemManager::batteryPercentageUpdated);
    ASSERT_TRUE(spy.isValid());

    systemManager->updateSystemStatus();

    ASSERT_EQ(spy.count(), 1);
    EXPECT_FLOAT_EQ(spy.takeFirst().at(0).toFloat(), 75.0f);
}

TEST_F(SystemManagerTest, UpdateSystemStatus_EmitsIpAddress)
{
    EXPECT_CALL(mockInfoProvider, getIpAddress()).WillOnce(Return(QString("192.168.1.100")));

    QSignalSpy spy(systemManager, &SystemManager::ipAddressUpdated);
    ASSERT_TRUE(spy.isValid());

    systemManager->updateSystemStatus();

    ASSERT_EQ(spy.count(), 1);
    EXPECT_EQ(spy.takeFirst().at(0).toString(), "192.168.1.100");
}
