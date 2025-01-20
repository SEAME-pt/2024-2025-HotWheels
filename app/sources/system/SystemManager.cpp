#include "SystemManager.hpp"

SystemManager::SystemManager(QObject *parent)
    : QObject(parent)
    , m_timeTimer(new QTimer(this))
    , m_statusTimer(new QTimer(this))
    , m_batteryController(new BatteryController("/dev/i2c-1", 0x41, this))
{
    QTimer::singleShot(0, this, &SystemManager::updateSystemStatus);

    // Update time every second
    connect(m_timeTimer, &QTimer::timeout, this, &SystemManager::updateTime);
    m_timeTimer->start(1000);

    // Update system status (WiFi, temperature, battery) every 5 seconds
    connect(m_statusTimer, &QTimer::timeout, this, &SystemManager::updateSystemStatus);
    m_statusTimer->start(5000);
}

void SystemManager::updateTime()
{
    QString currentTime = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    emit timeUpdated(currentTime);
}

void SystemManager::updateSystemStatus()
{
    // Fetch and emit WiFi status
    QString wifiName;
    QString wifiStatus = fetchWifiStatus(wifiName);
    emit wifiStatusUpdated(wifiStatus, wifiName);

    // Fetch and emit temperature
    QString temperature = fetchTemperature();
    emit temperatureUpdated(temperature);

    // Fetch and emit battery percentage
    float batteryPercentage = m_batteryController->getBatteryPercentage();
    emit batteryPercentageUpdated(batteryPercentage);
}

QString SystemManager::fetchWifiStatus(QString &wifiName) const
{
    QProcess process;
    process.start("nmcli", {"-t", "-f", "DEVICE,STATE,CONNECTION", "dev"});
    process.waitForFinished();

    QString output = process.readAllStandardOutput().trimmed();
    QStringList lines = output.split('\n');

    for (const QString &line : lines) {
        if (line.startsWith("wlan")) { // Assuming WiFi interface starts with 'wlan'
            QStringList parts = line.split(':');
            if (parts.size() >= 3) {
                QString state = parts[1];
                wifiName = parts[2]; // Extract connection name

                if (state == "connected") {
                    return "Connected";
                } else {
                    wifiName.clear();
                    return "Disconnected";
                }
            }
        }
    }
    wifiName.clear();
    return "No interface detected";
}

QString SystemManager::fetchTemperature() const
{
    QString tempFile = "/sys/class/hwmon/hwmon0/temp1_input";
    QFile tempInput(tempFile);
    if (tempInput.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&tempInput);
        QString tempStr = in.readLine().trimmed();
        tempInput.close();

        bool ok;
        double tempMillidegrees = tempStr.toDouble(&ok);
        if (ok) {
            return QString("%1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1);
        }
    }
    return "N/A";
}
