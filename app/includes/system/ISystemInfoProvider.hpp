#ifndef ISYSTEMINFOPROVIDER_HPP
#define ISYSTEMINFOPROVIDER_HPP

#include <QString>

class ISystemInfoProvider
{
public:
    virtual ~ISystemInfoProvider() = default;
    virtual QString getWifiStatus(QString &wifiName) const = 0;
    virtual QString getTemperature() const = 0;
    virtual QString getIpAddress() const = 0;
};

#endif // ISYSTEMINFOPROVIDER_HPP
