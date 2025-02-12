/*!
 * @file ClusterSettingsManager.hpp
 * @brief File containing the ClusterSettingsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details Definition of the ClusterSettingsManager class, which is responsible
 * for managing the cluster settings.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CLUSTERSETTINGSMANAGER_HPP
#define CLUSTERSETTINGSMANAGER_HPP

#include <QObject>
#include "enums.hpp"

/*!
 * @brief Class that manages the cluster settings.
 * @class ClusterSettingsManager
 */
class ClusterSettingsManager : public QObject
{
    Q_OBJECT

public:
    explicit ClusterSettingsManager(QObject *parent = nullptr);
    ~ClusterSettingsManager();

public slots:
    void toggleDrivingMode();
    void toggleClusterTheme();
    void toggleClusterMetrics();

signals:
    void drivingModeUpdated(DrivingMode newMode);
    void clusterThemeUpdated(ClusterTheme newTheme);
    void clusterMetricsUpdated(ClusterMetrics newMetrics);

private:
    DrivingMode m_drivingMode = DrivingMode::Manual;
    ClusterTheme m_clusterTheme = ClusterTheme::Dark;
    ClusterMetrics m_clusterMetrics = ClusterMetrics::Kilometers;

    void setDrivingMode(DrivingMode newMode);
    void setClusterTheme(ClusterTheme newTheme);
    void setClusterMetrics(ClusterMetrics newMetrics);
};

#endif // CLUSTERSETTINGSMANAGER_HPP
