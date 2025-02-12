/*!
 * @file test_BatteryController.cpp
 * @brief Unit tests for the BatteryController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the BatteryController class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "BatteryController.hpp"
#include "MockI2CController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

/*!
 * @class BatteryControllerTest
 * @brief Test fixture for testing the BatteryController class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class BatteryControllerTest : public ::testing::Test
{
protected:
    NiceMock<MockI2CController> mockI2C;
    BatteryController *batteryController;

    void SetUp() override { batteryController = new BatteryController(&mockI2C); }

    void TearDown() override { delete batteryController; }
};

/*!
 * @test Tests if the battery controller initializes correctly.
 * @brief Ensures that the battery controller initializes correctly.
 *
 * @details This test verifies that the battery controller initializes correctly.
 */
TEST_F(BatteryControllerTest, Initialization_CallsCalibration)
{
    EXPECT_CALL(mockI2C, writeRegister(0x05, 4096)).Times(1);
    delete batteryController; // Force constructor to be re-run
    batteryController = new BatteryController(&mockI2C);
}

/*!
 * @test Tests if the battery percentage is calculated correctly.
 * @brief Ensures that the battery percentage is calculated correctly.
 *
 * @details This test verifies that the battery percentage is calculated correctly.
 * The test uses known values for the bus voltage and shunt voltage to calculate
 * the expected battery percentage.
 * 
 * @see BatteryController::getBatteryPercentage
 */
TEST_F(BatteryControllerTest, GetBatteryPercentage_CorrectCalculation)
{
    EXPECT_CALL(mockI2C, readRegister(0x02)).WillOnce(Return(1000)); // Bus voltage raw value
    EXPECT_CALL(mockI2C, readRegister(0x01)).WillOnce(Return(100));  // Shunt voltage raw value

    float busVoltage = (1000 >> 3) * 0.004f;
    float shuntVoltage = 100 * 0.01f;
    float loadVoltage = busVoltage + shuntVoltage;
    float expectedPercentage = (loadVoltage - 6.0f) / 2.4f * 100.0f;

    if (expectedPercentage > 100.0f)
        expectedPercentage = 100.0f;
    if (expectedPercentage < 0.0f)
        expectedPercentage = 0.0f;

    float percentage = batteryController->getBatteryPercentage();
    EXPECT_NEAR(percentage, expectedPercentage, 0.1f);
}
