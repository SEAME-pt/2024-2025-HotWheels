#ifndef SYSTEMMANAGER_HPP
#define SYSTEMMANAGER_HPP

#include <QObject>
#include <QTimer>
#include "IBatteryController.hpp"
#include "ISystemCommandExecutor.hpp"
#include "ISystemInfoProvider.hpp"

class SystemManager : public QObject
{
    Q_OBJECT

public:
    explicit SystemManager(IBatteryController *batteryController = nullptr,
                           ISystemInfoProvider *systemInfoProvider = nullptr,
                           ISystemCommandExecutor *systemCommandExecutor = nullptr,
                           QObject *parent = nullptr);
    ~SystemManager();

    void initialize();
    void shutdown();

signals:
    void timeUpdated(const QString &currentDate,
                     const QString &currentTime,
                     const QString &currentDay);
    void wifiStatusUpdated(const QString &status, const QString &wifiName);
    void temperatureUpdated(const QString &temperature);
    void batteryPercentageUpdated(float batteryPercentage);
    void ipAddressUpdated(const QString &ipAddress);

public slots:
    void updateTime();
    void updateSystemStatus();

private:
    QTimer m_timeTimer;
    QTimer m_statusTimer;
    IBatteryController *m_batteryController;
    ISystemInfoProvider *m_systemInfoProvider;
    ISystemCommandExecutor *m_systemCommandExecutor;
    bool m_ownBatteryController;
    bool m_ownSystemInfoProvider;
    bool m_ownSystemCommandExecutor;
};

#endif // SYSTEMMANAGER_HPP
