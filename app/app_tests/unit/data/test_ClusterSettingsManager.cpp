/*!
 * @file test_ClusterSettingsManager.cpp
 * @brief Unit tests for the ClusterSettingsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains unit tests for the ClusterSettingsManager class, using
 * Google Test and Google Mock frameworks.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include <QSignalSpy>
#include "ClusterSettingsManager.hpp"
#include <gtest/gtest.h>

/*!
 * @class ClusterSettingsManagerTest
 * @brief Test fixture for testing the ClusterSettingsManager class.
 *
 * @details This class sets up the necessary objects and provides setup and
 * teardown methods for each test.
 */
class ClusterSettingsManagerTest : public ::testing::Test
{
protected:
    ClusterSettingsManager *clusterSettingsManager;

    void SetUp() override { clusterSettingsManager = new ClusterSettingsManager(); }

    void TearDown() override { delete clusterSettingsManager; }
};

/*!
 * @test Tests if the driving mode can be toggled.
 * @brief Ensures that the driving mode can be toggled between Manual and Automatic.
 *
 * @details This test verifies that the driving mode can be toggled between Manual
 * and Automatic.
 *
 * @see ClusterSettingsManager::toggleDrivingMode
 */
TEST_F(ClusterSettingsManagerTest, ToggleDrivingModeEmitsSignal)
{
    QSignalSpy drivingModeSpy(clusterSettingsManager, &ClusterSettingsManager::drivingModeUpdated);

    // Default mode is Manual, toggling should switch to Automatic
    clusterSettingsManager->toggleDrivingMode();
    ASSERT_EQ(drivingModeSpy.count(), 1);
    ASSERT_EQ(drivingModeSpy.takeFirst().at(0).value<DrivingMode>(), DrivingMode::Automatic);

    // Toggling again should switch back to Manual
    clusterSettingsManager->toggleDrivingMode();
    ASSERT_EQ(drivingModeSpy.count(), 1);
    ASSERT_EQ(drivingModeSpy.takeFirst().at(0).value<DrivingMode>(), DrivingMode::Manual);
}

/*!
 * @test Tests if the cluster theme can be toggled.
 * @brief Ensures that the cluster theme can be toggled between Dark and Light.
 *
 * @details This test verifies that the cluster theme can be toggled between Dark
 * and Light.
 *
 * @see ClusterSettingsManager::toggleClusterTheme
 */
TEST_F(ClusterSettingsManagerTest, ToggleClusterThemeEmitsSignal)
{
    QSignalSpy themeSpy(clusterSettingsManager, &ClusterSettingsManager::clusterThemeUpdated);

    // Default theme is Dark, toggling should switch to Light
    clusterSettingsManager->toggleClusterTheme();
    ASSERT_EQ(themeSpy.count(), 1);
    ASSERT_EQ(themeSpy.takeFirst().at(0).value<ClusterTheme>(), ClusterTheme::Light);

    // Toggling again should switch back to Dark
    clusterSettingsManager->toggleClusterTheme();
    ASSERT_EQ(themeSpy.count(), 1);
    ASSERT_EQ(themeSpy.takeFirst().at(0).value<ClusterTheme>(), ClusterTheme::Dark);
}

/*!
 * @test Tests if the cluster metrics can be toggled.
 * @brief Ensures that the cluster metrics can be toggled between Kilometers and Miles.
 *
 * @details This test verifies that the cluster metrics can be toggled between Kilometers
 * and Miles.
 *
 * @see ClusterSettingsManager::toggleClusterMetrics
 */
TEST_F(ClusterSettingsManagerTest, ToggleClusterMetricsEmitsSignal)
{
    QSignalSpy metricsSpy(clusterSettingsManager, &ClusterSettingsManager::clusterMetricsUpdated);

    // Default metrics is Kilometers, toggling should switch to Miles
    clusterSettingsManager->toggleClusterMetrics();
    ASSERT_EQ(metricsSpy.count(), 1);
    ASSERT_EQ(metricsSpy.takeFirst().at(0).value<ClusterMetrics>(), ClusterMetrics::Miles);

    // Toggling again should switch back to Kilometers
    clusterSettingsManager->toggleClusterMetrics();
    ASSERT_EQ(metricsSpy.count(), 1);
    ASSERT_EQ(metricsSpy.takeFirst().at(0).value<ClusterMetrics>(), ClusterMetrics::Kilometers);
}
