#ifndef CLUSTERSETTINGSMANAGER_HPP
#define CLUSTERSETTINGSMANAGER_HPP

#include <QObject>
#include "enums.hpp"

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
