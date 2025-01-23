#include "DisplayManager.hpp"
#include <QDebug>
#include <QPushButton>

DisplayManager::DisplayManager(Ui::CarManager *ui, QObject *parent)
    : QObject(parent)
    , m_ui(ui)
{
    // Ensure the labels are initialized
    if (!m_ui->speedLabel || !m_ui->rpmLabel || !m_ui->directionLabel || !m_ui->timeLabel
        || !m_ui->wifiLabel || !m_ui->temperatureLabel || !m_ui->batteryLabel) {
        qDebug() << "Error: Labels not initialized in the UI form!";
        return;
    }

    // Set initial values for the labels
    m_ui->speedLabel->setText("0");
    m_ui->rpmLabel->setText("0.00");
    m_ui->directionLabel->setText("D");
    m_ui->timeLabel->setText("--:--:--");
    m_ui->wifiLabel->setText("📶 Disconnected");
    m_ui->temperatureLabel->setText("🌡️ N/A");
    m_ui->batteryLabel->setText("--% 🔋");
    m_ui->speedMetricsLabel->setText("KM/H");
    m_ui->leftBlinkerLabel->setVisible(false);
    m_ui->rightBlinkerLabel->setVisible(false);
    m_ui->lowBatteryLabel->setVisible(false);

    // Directly connect button clicks to signals
    connect(m_ui->toggleDrivingModeButton,
            &QPushButton::clicked,
            this,
            &DisplayManager::drivingModeToggled);
    connect(m_ui->toggleMetricsButton,
            &QPushButton::clicked,
            this,
            &DisplayManager::clusterMetricsToggled);
}

void DisplayManager::updateCanBusData(float speed, int rpm)
{
    m_ui->speedLabel->setText(QString::number(static_cast<int>(speed)));
    m_ui->rpmLabel->setText(QString::number(static_cast<double>(rpm) / 1000, 'f', 2));
}

void DisplayManager::updateEngineData(CarDirection direction, int steeringAngle)
{
    QString directionText;
    switch (direction) {
    case CarDirection::Drive:
        directionText = "D";
        break;
    case CarDirection::Reverse:
        directionText = "R";
        break;
    case CarDirection::Stop:
    default:
        directionText = "D";
        break;
    }

    m_ui->directionLabel->setText(directionText);
    if (steeringAngle > 0) {
        m_ui->leftBlinkerLabel->setVisible(false);
        m_ui->rightBlinkerLabel->setVisible(true);
    } else if (steeringAngle < 0) {
        m_ui->rightBlinkerLabel->setVisible(false);
        m_ui->leftBlinkerLabel->setVisible(true);
    } else {
        m_ui->leftBlinkerLabel->setVisible(false);
        m_ui->rightBlinkerLabel->setVisible(false);
    }
}

void DisplayManager::updateSystemTime(const QString &currentDate,
                                      const QString &currentTime,
                                      const QString &currentDay)
{
    m_ui->dateLabel->setText(currentDate);
    m_ui->timeLabel->setText(currentTime);
    m_ui->weekDayLabel->setText(currentDay);
}

void DisplayManager::updateWifiStatus(const QString &status, const QString &wifiName)
{
    QString wifiDisplay = status;
    if (!wifiName.isEmpty()) {
        wifiDisplay += " (" + wifiName + ")";
    }
    m_ui->wifiLabel->setText("📶 " + wifiName);
}

void DisplayManager::updateTemperature(const QString &temperature)
{
    m_ui->temperatureLabel->setText("🌡️ " + temperature);
}

void DisplayManager::updateBatteryPercentage(float batteryPercentage)
{
    if (batteryPercentage < 20.0) {
        m_ui->lowBatteryLabel->setVisible(true);
    }
    m_ui->batteryLabel->setText(QString::number(batteryPercentage, 'f', 1) + "% "
                                + (batteryPercentage > 20.0 ? "🔋" : "🪫"));
}

void DisplayManager::updateMileage(double mileage)
{
    m_ui->mileageLabel->setText(QString::number(static_cast<int>(mileage)) + " m");
}

void DisplayManager::updateIpAddress(const QString &ipAddress)
{
    m_ui->ipAddressLabel->setText("IP " + ipAddress);
}

void DisplayManager::updateDrivingMode(DrivingMode newMode)
{
    QString modeText;
    switch (newMode) {
    case DrivingMode::Manual:
        modeText = "manual";
        break;
    case DrivingMode::Automatic:
        modeText = "automatic";
        break;
    }
    m_ui->drivingModeLabel->setText("Mode: " + modeText);
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
        metricsText = "km/h";
        break;
    case ClusterMetrics::Miles:
        metricsText = "mph";
        break;
    }
    m_ui->clusterMetricsLabel->setText("Metrics: " + metricsText);
    m_ui->speedMetricsLabel->setText(metricsText.toUpper());
}
