#include "SystemInfoUtility.hpp"
#include <QDebug>
#include <QFile>
#include <QProcess>
#include <QTextStream>

QString SystemInfoUtility::getWifiStatus()
{
    QProcess process;
    QString program = "nmcli";
    QStringList arguments = {"-t", "-f", "DEVICE,STATE,CONNECTION", "dev"};
    process.start(program, arguments);
    process.waitForFinished();

    QString output = process.readAllStandardOutput().trimmed();
    QStringList lines = output.split('\n');

    for (const QString &line : lines) {
        if (line.startsWith("wlan")) { // Assuming your WiFi interface starts with 'wlan'
            QStringList parts = line.split(':');
            if (parts.size() >= 3) {
                QString state = parts[1];
                QString connection = parts[2];

                if (state == "connected") {
                    return QString("Connected to %1").arg(connection);
                } else {
                    return "WiFi: Disconnected";
                }
            }
        }
    }
    return "WiFi: No interface detected";
}

QString SystemInfoUtility::getTemperature()
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
            return QString("System Temperature: %1°C").arg(tempMillidegrees / 1000.0, 0, 'f', 1);
        }
    }
    return "Temperature: N/A";
}
