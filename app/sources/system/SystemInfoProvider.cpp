#include "SystemInfoProvider.hpp"
#include <QDebug>
#include "SystemCommandExecutor.hpp"

SystemInfoProvider::SystemInfoProvider(ISystemCommandExecutor *executor)
    : m_executor(executor ? executor : new SystemCommandExecutor())
    , m_ownExecutor(executor == nullptr)
{}

SystemInfoProvider::~SystemInfoProvider()
{
    if (m_ownExecutor)
        delete m_executor;
}

QString SystemInfoProvider::getWifiStatus(QString &wifiName) const
{
    QString output = m_executor->executeCommand("nmcli -t -f DEVICE,STATE,CONNECTION dev");
    QStringList lines = output.split('\n');

    for (const QString &line : lines) {
        if (line.startsWith("wlan")) {
            QStringList parts = line.split(':');
            if (parts.size() >= 3) {
                wifiName = parts[2];
                return (parts[1] == "connected") ? "Connected" : "Disconnected";
            }
        }
    }
    wifiName.clear();
    return "No interface detected";
}

QString SystemInfoProvider::getTemperature() const
{
    QString tempStr = m_executor->readFile("/sys/class/hwmon/hwmon0/temp1_input").trimmed();

    bool ok;
    double tempMillidegrees = tempStr.toDouble(&ok);
    return ok ? QString("%1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1) : "N/A";
}

QString SystemInfoProvider::getIpAddress() const
{
    QString output = m_executor->executeCommand(
        "sh -c \"ip -4 addr show wlan0 | grep -oP '(?<=inet\\s)\\d+\\.\\d+\\.\\d+\\.\\d+'\"");

    return output.trimmed().isEmpty() ? "No IP address" : output.trimmed();
}
