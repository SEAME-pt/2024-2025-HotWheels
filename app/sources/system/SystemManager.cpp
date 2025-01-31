#include "SystemManager.hpp"
#include <QDateTime>
#include <QDebug>
#include "BatteryController.hpp"
#include "SystemCommandExecutor.hpp"
#include "SystemInfoProvider.hpp"

SystemManager::SystemManager(IBatteryController *batteryController,
                             ISystemInfoProvider *systemInfoProvider,
                             ISystemCommandExecutor *systemCommandExecutor,
                             QObject *parent)
    : QObject(parent)
    , m_batteryController(batteryController ? batteryController : new BatteryController())
    , m_systemInfoProvider(systemInfoProvider ? systemInfoProvider : new SystemInfoProvider())
    , m_systemCommandExecutor(systemCommandExecutor ? systemCommandExecutor
                                                    : new SystemCommandExecutor())
    , m_ownBatteryController(batteryController == nullptr)
    , m_ownSystemInfoProvider(systemInfoProvider == nullptr)
    , m_ownSystemCommandExecutor(systemCommandExecutor == nullptr)
{}

SystemManager::~SystemManager()
{
    shutdown();
    if (m_ownBatteryController)
        delete m_batteryController;
    if (m_ownSystemInfoProvider)
        delete m_systemInfoProvider;
    if (m_ownSystemCommandExecutor)
        delete m_systemCommandExecutor;
}

void SystemManager::initialize()
{
    connect(&m_timeTimer, &QTimer::timeout, this, &SystemManager::updateTime);
    connect(&m_statusTimer, &QTimer::timeout, this, &SystemManager::updateSystemStatus);
    m_timeTimer.start(1000);
    m_statusTimer.start(5000);
    updateSystemStatus();
}

void SystemManager::shutdown()
{
    m_timeTimer.stop();
    m_statusTimer.stop();
}

void SystemManager::updateTime()
{
    QDateTime currentDateTime = QDateTime::currentDateTime();
    emit timeUpdated(currentDateTime.toString("dd-MM-yy"),
                     currentDateTime.toString("HH:mm"),
                     currentDateTime.toString("dddd"));
}

void SystemManager::updateSystemStatus()
{
    QString wifiName;
    emit wifiStatusUpdated(m_systemInfoProvider->getWifiStatus(wifiName), wifiName);
    emit temperatureUpdated(m_systemInfoProvider->getTemperature());
    emit batteryPercentageUpdated(m_batteryController->getBatteryPercentage());
    emit ipAddressUpdated(m_systemInfoProvider->getIpAddress());
}
