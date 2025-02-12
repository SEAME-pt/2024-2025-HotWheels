/*!
 * @file ClusterSettingsManager.cpp
 * @brief Implementation of the ClusterSettingsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the ClusterSettingsManager
 * class, which manages the settings of the cluster.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ClusterSettingsManager.hpp"

/*!
 * @brief Construct a new ClusterSettingsManager::ClusterSettingsManager object
 * @param parent The parent QObject.
 * @details This constructor initializes the ClusterSettingsManager object.
 */
ClusterSettingsManager::ClusterSettingsManager(QObject *parent)
    : QObject(parent)
{}

/*!
 * @brief Destroy the ClusterSettingsManager::ClusterSettingsManager object
 * @details This destructor cleans up the resources used by the
 * ClusterSettingsManager.
 */
ClusterSettingsManager::~ClusterSettingsManager() {}

/*!
 * @brief Get the driving mode.
 * @returns The driving mode.
 * @details This function returns the current driving mode.
 */
void ClusterSettingsManager::setDrivingMode(DrivingMode newMode)
{
    if (m_drivingMode != newMode) {
        m_drivingMode = newMode;
        emit drivingModeUpdated(newMode);
    }
}

/*!
 * @brief Toggle the driving mode.
 * @details This function toggles the driving mode between manual and automatic.
 */
void ClusterSettingsManager::toggleDrivingMode()
{
    if (m_drivingMode == DrivingMode::Manual) {
        setDrivingMode(DrivingMode::Automatic);
    } else {
        setDrivingMode(DrivingMode::Manual);
    }
}

/*!
 * @brief Get the cluster theme.
 * @returns The cluster theme.
 * @details This function returns the current cluster theme.
 */
void ClusterSettingsManager::setClusterTheme(ClusterTheme newTheme)
{
    if (m_clusterTheme != newTheme) {
        m_clusterTheme = newTheme;
        emit clusterThemeUpdated(newTheme);
    }
}

/*!
 * @brief Toggle the cluster theme.
 * @details This function toggles the cluster theme between light and dark.
 */
void ClusterSettingsManager::toggleClusterTheme()
{
    if (m_clusterTheme == ClusterTheme::Dark) {
        setClusterTheme(ClusterTheme::Light);
    } else {
        setClusterTheme(ClusterTheme::Dark);
    }
}

/*!
 * @brief Get the cluster metrics.
 * @returns The cluster metrics.
 * @details This function returns the current cluster metrics.
 */
void ClusterSettingsManager::setClusterMetrics(ClusterMetrics newMetrics)
{
    if (m_clusterMetrics != newMetrics) {
        m_clusterMetrics = newMetrics;
        emit clusterMetricsUpdated(newMetrics);
    }
}

/*!
 * @brief Toggle the cluster metrics.
 * @details This function toggles the cluster metrics between kilometers and miles.
 */
void ClusterSettingsManager::toggleClusterMetrics()
{
    if (m_clusterMetrics == ClusterMetrics::Kilometers) {
        setClusterMetrics(ClusterMetrics::Miles);
    } else {
        setClusterMetrics(ClusterMetrics::Kilometers);
    }
}
