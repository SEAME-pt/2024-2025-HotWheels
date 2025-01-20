#ifndef SYSTEMMANAGER_HPP
#define SYSTEMMANAGER_HPP

#include <QDateTime>
#include <QFile>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QTextStream>
#include <QTimer>
#include "BatteryController.hpp"

class SystemManager : public QObject
{
    Q_OBJECT

public:
    explicit SystemManager(QObject *parent = nullptr);

signals:
    void timeUpdated(const QString &currentDate,
                     const QString &currentTime,
                     const QString &currentDay);
    void wifiStatusUpdated(const QString &status, const QString &wifiName);
    void temperatureUpdated(const QString &temperature);
    void batteryPercentageUpdated(float batteryPercentage);
    void ipAddressUpdated(const QString &ipAddress);

private slots:
    void updateTime();
    void updateSystemStatus();

private:
    QString fetchWifiStatus(QString &wifiName) const;
    QString fetchTemperature() const;
    QString fetchIpAddress() const;

    QTimer *m_timeTimer;
    QTimer *m_statusTimer;
    BatteryController *m_batteryController;
};

#endif // SYSTEMMANAGER_HPP
