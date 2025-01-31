#include "SystemInfoProvider.hpp"
#include <QFile>
#include <QProcess>
#include <QTextStream>

QString SystemInfoProvider::getWifiStatus(QString &wifiName) const
{
    QProcess process;
    process.start("nmcli", {"-t", "-f", "DEVICE,STATE,CONNECTION", "dev"});
    process.waitForFinished();
    QString output = process.readAllStandardOutput().trimmed();

    for (const QString &line : output.split('\n')) {
        if (line.startsWith("wlan")) {
            QStringList parts = line.split(':');
            if (parts.size() >= 3) {
                wifiName = parts[2];
                return parts[1] == "connected" ? "Connected" : "Disconnected";
            }
        }
    }
    wifiName.clear();
    return "No interface detected";
}

QString SystemInfoProvider::getTemperature() const
{
    QFile tempFile("/sys/class/hwmon/hwmon0/temp1_input");
    if (tempFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&tempFile);
        QString tempStr = in.readLine().trimmed();
        bool ok;
        double tempMillidegrees = tempStr.toDouble(&ok);
        return ok ? QString("%1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1) : "N/A";
    }
    return "N/A";
}

QString SystemInfoProvider::getIpAddress() const
{
    QProcess process;
    process.start("sh",
                  {"-c", "ip addr show wlan0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1"});
    process.waitForFinished();
    return process.readAllStandardOutput().trimmed().isEmpty()
               ? "No IP address"
               : process.readAllStandardOutput().trimmed();
}
