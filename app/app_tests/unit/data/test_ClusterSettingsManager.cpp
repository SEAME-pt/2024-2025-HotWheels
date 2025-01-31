#include <QSignalSpy>
#include "ClusterSettingsManager.hpp"
#include <gtest/gtest.h>

class ClusterSettingsManagerTest : public ::testing::Test
{
protected:
    ClusterSettingsManager *clusterSettingsManager;

    void SetUp() override { clusterSettingsManager = new ClusterSettingsManager(); }

    void TearDown() override { delete clusterSettingsManager; }
};

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
