#include "SystemManager.hpp"
#include <QDateTime>
#include <QDebug>
#include "BatteryController.hpp"
#include "SystemInfoProvider.hpp"

SystemManager::SystemManager(IBatteryController *batteryController,
                             ISystemInfoProvider *systemInfoProvider,
                             QObject *parent)
    : QObject(parent)
    , m_batteryController(batteryController ? batteryController
                                            : new BatteryController("/dev/i2c-1", 0x41, this))
    , m_systemInfoProvider(systemInfoProvider ? systemInfoProvider : new SystemInfoProvider())
    , m_ownBatteryController(batteryController == nullptr)
    , m_ownSystemInfoProvider(systemInfoProvider == nullptr)
{}

SystemManager::~SystemManager()
{
    shutdown();
    if (m_ownBatteryController)
        delete m_batteryController;
    if (m_ownSystemInfoProvider)
        delete m_systemInfoProvider;
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

void SystemManager::updateSystemStatus()
{
    QString wifiName;
    emit wifiStatusUpdated(m_systemInfoProvider->getWifiStatus(wifiName), wifiName);
    emit temperatureUpdated(m_systemInfoProvider->getTemperature());
    emit batteryPercentageUpdated(m_batteryController->getBatteryPercentage());
    emit ipAddressUpdated(m_systemInfoProvider->getIpAddress());
}
