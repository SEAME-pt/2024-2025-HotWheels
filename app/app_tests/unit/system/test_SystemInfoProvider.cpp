#include "MockSystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

class SystemInfoProviderTest : public ::testing::Test
{
protected:
    NiceMock<MockSystemCommandExecutor> mockExecutor;
    SystemInfoProvider *infoProvider;

    void SetUp() override { infoProvider = new SystemInfoProvider(&mockExecutor); }

    void TearDown() override { delete infoProvider; }
};

TEST_F(SystemInfoProviderTest, GetWifiStatus_Connected)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("wlan0:connected:MyWiFi")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "Connected");
    EXPECT_EQ(wifiName, "MyWiFi");
}

TEST_F(SystemInfoProviderTest, GetWifiStatus_Disconnected)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("wlan0:disconnected:")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "Disconnected");
    EXPECT_TRUE(wifiName.isEmpty());
}

TEST_F(SystemInfoProviderTest, GetWifiStatus_NoInterface)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("eth0:connected:Ethernet")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "No interface detected");
    EXPECT_TRUE(wifiName.isEmpty());
}

TEST_F(SystemInfoProviderTest, GetTemperature_ValidReading)
{
    EXPECT_CALL(mockExecutor, readFile(QString("/sys/class/hwmon/hwmon0/temp1_input")))
        .WillOnce(Return(QString("45000")));

    QString temperature = infoProvider->getTemperature();

    EXPECT_EQ(temperature, "45.0°C");
}

TEST_F(SystemInfoProviderTest, GetTemperature_InvalidReading)
{
    EXPECT_CALL(mockExecutor, readFile(QString("/sys/class/hwmon/hwmon0/temp1_input")))
        .WillOnce(Return(QString("INVALID")));

    QString temperature = infoProvider->getTemperature();

    EXPECT_EQ(temperature, "N/A");
}

TEST_F(SystemInfoProviderTest, GetIpAddress_Valid)
{
    EXPECT_CALL(
        mockExecutor,
        executeCommand(QString(
            "sh -c \"ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+\\.\\d+\\.\\d+\\.\\d+'\"")))
        .WillOnce(Return(QString("192.168.1.100")));

    QString ipAddress = infoProvider->getIpAddress();

    EXPECT_EQ(ipAddress, "192.168.1.100");
}

TEST_F(SystemInfoProviderTest, GetIpAddress_NoIP)
{
    EXPECT_CALL(
        mockExecutor,
        executeCommand(QString(
            "sh -c \"ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+\\.\\d+\\.\\d+\\.\\d+'\"")))
        .WillOnce(Return(QString("")));

    QString ipAddress = infoProvider->getIpAddress();

    EXPECT_EQ(ipAddress, "No IP address");
}
