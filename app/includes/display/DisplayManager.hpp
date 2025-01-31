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
    void updateCanBusData(float speed, int rpm);
    void updateEngineData(CarDirection direction, int steeringAngle);
    void updateSystemTime(const QString &currentDate,
                          const QString &currentTime,
                          const QString &currentDay);
    void updateWifiStatus(const QString &status, const QString &wifiName);
    void updateTemperature(const QString &temperature);
    void updateBatteryPercentage(float batteryPercentage);
    void updateIpAddress(const QString &ipAddress);
    void updateMileage(double mileage);
    void updateDrivingMode(DrivingMode newMode);

    void updateClusterTheme(ClusterTheme newTheme);
    void updateClusterMetrics(ClusterMetrics newMetrics);

signals:
    void drivingModeToggled();
    void clusterThemeToggled();
    void clusterMetricsToggled();

private:
    Ui::CarManager *m_ui;
};

#endif // DISPLAYMANAGER_HPP
