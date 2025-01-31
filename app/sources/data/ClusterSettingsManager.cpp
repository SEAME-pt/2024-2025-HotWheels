#include "ClusterSettingsManager.hpp"

ClusterSettingsManager::ClusterSettingsManager(QObject *parent)
    : QObject(parent)
{}

ClusterSettingsManager::~ClusterSettingsManager() {}

// Driving Mode Handling
void ClusterSettingsManager::setDrivingMode(DrivingMode newMode)
{
    if (m_drivingMode != newMode) {
        m_drivingMode = newMode;
        emit drivingModeUpdated(newMode);
    }
}

void ClusterSettingsManager::toggleDrivingMode()
{
    if (m_drivingMode == DrivingMode::Manual) {
        setDrivingMode(DrivingMode::Automatic);
    } else {
        setDrivingMode(DrivingMode::Manual);
    }
}

// Cluster Theme Handling
void ClusterSettingsManager::setClusterTheme(ClusterTheme newTheme)
{
    if (m_clusterTheme != newTheme) {
        m_clusterTheme = newTheme;
        emit clusterThemeUpdated(newTheme);
    }
}

void ClusterSettingsManager::toggleClusterTheme()
{
    if (m_clusterTheme == ClusterTheme::Dark) {
        setClusterTheme(ClusterTheme::Light);
    } else {
        setClusterTheme(ClusterTheme::Dark);
    }
}

// Cluster Metrics Handling
void ClusterSettingsManager::setClusterMetrics(ClusterMetrics newMetrics)
{
    if (m_clusterMetrics != newMetrics) {
        m_clusterMetrics = newMetrics;
        emit clusterMetricsUpdated(newMetrics);
    }
}

void ClusterSettingsManager::toggleClusterMetrics()
{
    if (m_clusterMetrics == ClusterMetrics::Kilometers) {
        setClusterMetrics(ClusterMetrics::Miles);
    } else {
        setClusterMetrics(ClusterMetrics::Kilometers);
    }
}
