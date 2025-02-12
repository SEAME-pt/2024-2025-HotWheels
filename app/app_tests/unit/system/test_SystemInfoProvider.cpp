/*!
 * @file test_SystemInfoProvider.cpp
 * @brief Unit tests for the SystemInfoProvider class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the SystemInfoProvider class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "MockSystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

/*!
 * @class SystemInfoProviderTest
 * @brief Test fixture for testing the SystemInfoProvider class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class SystemInfoProviderTest : public ::testing::Test
{
protected:
    NiceMock<MockSystemCommandExecutor> mockExecutor;
    SystemInfoProvider *infoProvider;

    void SetUp() override { infoProvider = new SystemInfoProvider(&mockExecutor); }

    void TearDown() override { delete infoProvider; }
};

/*!
 * @test Tests if the wifi status is correctly retrieved.
 * @brief Ensures that the wifi status is correctly retrieved.
 * @details This test verifies that the wifi status is correctly retrieved.
 * The wifi status should be "Connected" and the wifi name should be "MyWiFi".
 *
 * @see SystemInfoProvider::getWifiStatus
*/
TEST_F(SystemInfoProviderTest, GetWifiStatus_Connected)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("wlan0:connected:MyWiFi")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "Connected");
    EXPECT_EQ(wifiName, "MyWiFi");
}

/*!
 * @test Tests if the wifi status is correctly retrieved when disconnected.
 * @brief Ensures that the wifi status is correctly retrieved when disconnected.
 * @details This test verifies that the wifi status is correctly retrieved when
 * disconnected. The wifi status should be "Disconnected" and the wifi name should
 * be empty.
 *
 * @see SystemInfoProvider::getWifiStatus
*/
TEST_F(SystemInfoProviderTest, GetWifiStatus_Disconnected)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("wlan0:disconnected:")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "Disconnected");
    EXPECT_TRUE(wifiName.isEmpty());
}

/*!
 * @test Tests if the wifi status is correctly retrieved when no interface is detected.
 * @brief Ensures that the wifi status is correctly retrieved when no interface is detected.
 * @details This test verifies that the wifi status is correctly retrieved when no
 * interface is detected. The wifi status should be "No interface detected" and the
 * wifi name should be empty.
 *
 * @see SystemInfoProvider::getWifiStatus
*/
TEST_F(SystemInfoProviderTest, GetWifiStatus_NoInterface)
{
    QString wifiName;
    EXPECT_CALL(mockExecutor, executeCommand(QString("nmcli -t -f DEVICE,STATE,CONNECTION dev")))
        .WillOnce(Return(QString("eth0:connected:Ethernet")));

    QString status = infoProvider->getWifiStatus(wifiName);

    EXPECT_EQ(status, "No interface detected");
    EXPECT_TRUE(wifiName.isEmpty());
}

/*!
 * @test Tests if the wifi status is correctly retrieved when the wifi name is empty.
 * @brief Ensures that the wifi status is correctly retrieved when the wifi name is empty.
 * @details This test verifies that the wifi status is correctly retrieved when the
 * wifi name is empty. The wifi status should be "Connected" and the wifi name should
 * be empty.
 *
 * @see SystemInfoProvider::getWifiStatus
*/
TEST_F(SystemInfoProviderTest, GetTemperature_ValidReading)
{
    EXPECT_CALL(mockExecutor, readFile(QString("/sys/class/hwmon/hwmon0/temp1_input")))
        .WillOnce(Return(QString("45000")));

    QString temperature = infoProvider->getTemperature();

    EXPECT_EQ(temperature, "45.0°C");
}

/*!
 * @test Tests if the temperature is correctly retrieved when the reading is invalid.
 * @brief Ensures that the temperature is correctly retrieved when the reading is invalid.
 * @details This test verifies that the temperature is correctly retrieved when the
 * reading is invalid. The temperature should be "N/A".
 *
 * @see SystemInfoProvider::getTemperature
*/
TEST_F(SystemInfoProviderTest, GetTemperature_InvalidReading)
{
    EXPECT_CALL(mockExecutor, readFile(QString("/sys/class/hwmon/hwmon0/temp1_input")))
        .WillOnce(Return(QString("INVALID")));

    QString temperature = infoProvider->getTemperature();

    EXPECT_EQ(temperature, "N/A");
}

/*!
 * @test Tests if the IP address is correctly retrieved.
 * @brief Ensures that the IP address is correctly retrieved.
 * @details This test verifies that the IP address is correctly retrieved. The IP
 * address should be "
 * 
 * @see SystemInfoProvider::getIpAddress
 */
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

/*!
 * @test Tests if the IP address is correctly retrieved when no IP address is found.
 * @brief Ensures that the IP address is correctly retrieved when no IP address is found.
 * @details This test verifies that the IP address is correctly retrieved when no IP
 * address is found. The IP address should be "No IP address".
 *
 * @see SystemInfoProvider::getIpAddress
 */
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
