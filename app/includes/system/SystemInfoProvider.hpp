#ifndef SYSTEMINFOPROVIDER_HPP
#define SYSTEMINFOPROVIDER_HPP

#include "ISystemInfoProvider.hpp"

class SystemInfoProvider : public ISystemInfoProvider
{
public:
    QString getWifiStatus(QString &wifiName) const override;
    QString getTemperature() const override;
    QString getIpAddress() const override;
};

#endif // SYSTEMINFOPROVIDER_HPP
