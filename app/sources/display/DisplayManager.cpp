#include "DisplayManager.hpp"
#include <QDebug>
#include <QPushButton>

DisplayManager::DisplayManager(Ui::CarManager *ui, QObject *parent)
    : QObject(parent)
    , m_ui(ui)
{
    // Ensure the labels are initialized
    if (!m_ui->speedLabel || !m_ui->rpmLabel || !m_ui->directionLabel || !m_ui->steeringLabel
        || !m_ui->timeLabel || !m_ui->wifiLabel || !m_ui->temperatureLabel || !m_ui->batteryLabel) {
        qDebug() << "Error: Labels not initialized in the UI form!";
        return;
    }

    // Set initial values for the labels
    m_ui->speedLabel->setText("Speed: 0.00 km/h");
    m_ui->rpmLabel->setText("RPM: 0");
    m_ui->directionLabel->setText("Direction: Stop");
    m_ui->steeringLabel->setText("Steering: 0°");
    m_ui->timeLabel->setText("Time: --:--:--");
    m_ui->wifiLabel->setText("WiFi: Disconnected");
    m_ui->temperatureLabel->setText("Temperature: N/A");
    m_ui->batteryLabel->setText("Battery: --%");

    // Directly connect button clicks to signals
    connect(m_ui->toggleDrivingModeButton,
            &QPushButton::clicked,
            this,
            &DisplayManager::drivingModeToggled);
    connect(m_ui->toggleThemeButton,
            &QPushButton::clicked,
            this,
            &DisplayManager::clusterThemeToggled);
    connect(m_ui->toggleMetricsButton,
            &QPushButton::clicked,
            this,
            &DisplayManager::clusterMetricsToggled);
}

void DisplayManager::updateCanBusData(float speed, int rpm, ClusterMetrics currentMetrics)
{
    QString speedMetricsLabel = currentMetrics == ClusterMetrics::Kilometers ? " km/h" : "m/h";
    m_ui->speedLabel->setText("Speed: " + QString::number(speed, 'f', 2) + " " + speedMetricsLabel);
    m_ui->rpmLabel->setText("RPM: " + QString::number(rpm));
}

void DisplayManager::updateEngineData(CarDirection direction, int steeringAngle)
{
    QString directionText;
    switch (direction) {
    case CarDirection::Drive:
        directionText = "Drive";
        break;
    case CarDirection::Reverse:
        directionText = "Reverse";
        break;
    case CarDirection::Stop:
    default:
        directionText = "Stop";
        break;
    }

    m_ui->directionLabel->setText("Direction: " + directionText);
    m_ui->steeringLabel->setText("Steering: " + QString::number(steeringAngle) + "°");
}

void DisplayManager::updateSystemTime(const QString &currentDate,
                                      const QString &currentTime,
                                      const QString &currentDay)
{
    m_ui->timeLabel->setText("Date: " + currentDate + " | Day: " + currentDay
                             + " | Time: " + currentTime);
}

void DisplayManager::updateWifiStatus(const QString &status, const QString &wifiName)
{
    QString wifiDisplay = status;
    if (!wifiName.isEmpty()) {
        wifiDisplay += " (" + wifiName + ")";
    }
    m_ui->wifiLabel->setText("WiFi: " + wifiDisplay);
}

void DisplayManager::updateTemperature(const QString &temperature)
{
    m_ui->temperatureLabel->setText("Temperature: " + temperature);
}

void DisplayManager::updateBatteryPercentage(float batteryPercentage)
{
    m_ui->batteryLabel->setText("Battery: " + QString::number(batteryPercentage, 'f', 1) + "%");
}

void DisplayManager::updateIpAddress(const QString &ipAddress)
{
    m_ui->ipAddressLabel->setText("IP: " + ipAddress);
}

void DisplayManager::updateDrivingMode(DrivingMode newMode)
{
    QString modeText;
    switch (newMode) {
    case DrivingMode::Manual:
        modeText = "Manual";
        break;
    case DrivingMode::Automatic:
        modeText = "Automatic";
        break;
    }
    m_ui->drivingModeLabel->setText("Driving Mode: " + modeText);
}

void DisplayManager::updateClusterTheme(ClusterTheme newTheme)
{
    QString themeText;
    switch (newTheme) {
    case ClusterTheme::Dark:
        themeText = "Dark";
        break;
    case ClusterTheme::Light:
        themeText = "Light";
        break;
    }
    m_ui->clusterThemeLabel->setText("Theme: " + themeText);
}

void DisplayManager::updateClusterMetrics(ClusterMetrics newMetrics)
{
    QString metricsText;
    switch (newMetrics) {
    case ClusterMetrics::Kilometers:
        metricsText = "Kilometers";
        break;
    case ClusterMetrics::Miles:
        metricsText = "Miles";
        break;
    }
    m_ui->clusterMetricsLabel->setText("Metrics: " + metricsText);
}
