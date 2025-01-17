#include "DisplayManager.hpp"
#include "CircularMeterSetup.hpp"
#include "SystemInfoUtility.hpp"

#include <QDateTime>

DisplayManager::DisplayManager(QWidget *parent,
                               QWidget *speedParentWidget,
                               MeterController *speedController,
                               QWidget *rpmParentWidget,
                               MeterController *rpmController,
                               QLabel *systemInfoLabel)
    : QObject(parent)
    , m_speedController(speedController)
    , m_speedWidget(new QQuickWidget())
    , m_speedParentWidget(speedParentWidget)
    , m_rpmController(rpmController)
    , m_rpmWidget(new QQuickWidget())
    , m_rpmParentWidget(rpmParentWidget)
    , m_systemInfoLabel(systemInfoLabel)
    , m_timeUpdateTimer(new QTimer(this))
{
    SystemInfoUtility::printClassInfo("DisplayManager", SystemInfoUtility::InfoType::CreBegin);
    setupTimeLabel();
    setupWidgets();
    SystemInfoUtility::printClassInfo("DisplayManager", SystemInfoUtility::InfoType::CreEnd);
}

DisplayManager::~DisplayManager()
{
    SystemInfoUtility::printClassInfo("DisplayManager", SystemInfoUtility::InfoType::DesBegin);
    delete m_speedWidget;
    delete m_rpmWidget;
    SystemInfoUtility::printClassInfo("DisplayManager", SystemInfoUtility::InfoType::DesEnd);
}

void DisplayManager::setupWidgets()
{
    CircularMeterSetup::setupQuickWidget(m_speedWidget,
                                         m_speedParentWidget,
                                         m_speedController,
                                         "#000957",
                                         1,
                                         "km",
                                         96);

    CircularMeterSetup::setupQuickWidget(m_rpmWidget,
                                         m_rpmParentWidget,
                                         m_rpmController,
                                         "#000957",
                                         100,
                                         "rpm",
                                         64);
}

void DisplayManager::setupTimeLabel()
{
    updateStatusDisplay();

    connect(m_timeUpdateTimer, &QTimer::timeout, this, &DisplayManager::updateStatusDisplay);
    m_timeUpdateTimer->start(1000);
}

void DisplayManager::updateStatusDisplay()
{
    QString currentTime = QDateTime::currentDateTime().toString("hh:mm:ss");
    QString wifiStatus = SystemInfoUtility::getWifiStatus();
    QString temperature = SystemInfoUtility::getTemperature();

    QString statusText = QString("Time: %1\nWiFi: %2\n%3")
                             .arg(currentTime)
                             .arg(wifiStatus)
                             .arg(temperature);

    m_systemInfoLabel->setText(statusText);
}
