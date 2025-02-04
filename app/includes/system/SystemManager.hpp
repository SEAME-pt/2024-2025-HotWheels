/*!
 * @file SystemManager.hpp
 * @brief Definition of the SystemManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the SystemManager class, which
 * is used to manage the system time, status, and battery.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SYSTEMMANAGER_HPP
#define SYSTEMMANAGER_HPP

#include <QObject>
#include <QTimer>

#include "IBatteryController.hpp"
#include "ISystemCommandExecutor.hpp"
#include "ISystemInfoProvider.hpp"

/*!
 * @brief Class that manages the system time, status, and battery.
 * @class SystemManager inherits from QObject
 */
class SystemManager : public QObject {
  Q_OBJECT

public:
    explicit SystemManager(IBatteryController *batteryController = nullptr,
                           ISystemInfoProvider *systemInfoProvider = nullptr,
                           ISystemCommandExecutor *systemCommandExecutor = nullptr,
                           QObject *parent = nullptr);
    ~SystemManager();

    void initialize();
    void shutdown();

    QTimer &getTimeTimer() { return this->m_timeTimer; };
    QTimer &getStatusTimer() { return this->m_statusTimer; };

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
