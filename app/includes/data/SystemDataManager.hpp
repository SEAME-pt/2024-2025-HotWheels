#ifndef SYSTEMDATAMANAGER_HPP
#define SYSTEMDATAMANAGER_HPP

#include <QObject>
#include <QString>

class SystemDataManager : public QObject
{
    Q_OBJECT

public:
    explicit SystemDataManager(QObject *parent = nullptr);
    ~SystemDataManager();

public slots:
    void handleTimeData(const QString &currentDate,
                        const QString &currentTime,
                        const QString &currentDay);
    void handleWifiData(const QString &status, const QString &wifiName);
    void handleTemperatureData(const QString &temperature);
    void handleIpAddressData(const QString &ipAddress);
    void handleBatteryPercentage(float batteryPercentage);

signals:
    void systemTimeUpdated(const QString &currentDate,
                           const QString &currentTime,
                           const QString &currentDay);
    void systemWifiUpdated(const QString &status, const QString &wifiName);
    void systemTemperatureUpdated(const QString &temperature);
    void ipAddressUpdated(const QString &ipAddress);
    void batteryPercentageUpdated(float batteryPercentage);

private:
    QString m_time = "";
    QString m_wifiName = "";
    QString m_wifiStatus = "";
    QString m_temperature = "";
    QString m_ipAddress = "";
    float m_batteryPercentage = -1.0f;
};

#endif // SYSTEMDATAMANAGER_HPP
