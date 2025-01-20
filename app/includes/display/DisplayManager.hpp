#ifndef DISPLAYMANAGER_HPP
#define DISPLAYMANAGER_HPP

#include <QObject>
#include <QString>
#include "enums.hpp"
#include "ui_CarManager.h"

class DisplayManager : public QObject
{
    Q_OBJECT

public:
    explicit DisplayManager(Ui::CarManager *ui, QObject *parent = nullptr);

public slots:
    // CAN
    void updateCanBusData(float speed, int rpm, ClusterMetrics currentMetrics);

    // Engine
    void updateEngineData(CarDirection direction, int steeringAngle);

    // System
    void updateSystemTime(const QString &currentDate,
                          const QString &currentTime,
                          const QString &currentDay);
    void updateWifiStatus(const QString &status, const QString &wifiName);
    void updateTemperature(const QString &temperature);
    void updateBatteryPercentage(float batteryPercentage);
    void updateIpAddress(const QString &ipAddress);

    // Settings
    void updateDrivingMode(DrivingMode newMode);
    void updateClusterTheme(ClusterTheme newTheme);
    void updateClusterMetrics(ClusterMetrics newMetrics);

signals:
    // Signals to DataManager
    void drivingModeToggled();
    void clusterThemeToggled();
    void clusterMetricsToggled();

private:
    Ui::CarManager *m_ui;
};

#endif // DISPLAYMANAGER_HPP
