#ifndef SYSTEMINFOPROVIDER_HPP
#define SYSTEMINFOPROVIDER_HPP

#include "ISystemCommandExecutor.hpp"
#include "ISystemInfoProvider.hpp"

class SystemInfoProvider : public ISystemInfoProvider
{
public:
    explicit SystemInfoProvider(ISystemCommandExecutor *executor = nullptr);
    ~SystemInfoProvider() override;

    QString getWifiStatus(QString &wifiName) const override;
    QString getTemperature() const override;
    QString getIpAddress() const override;

private:
    ISystemCommandExecutor *m_executor;
    bool m_ownExecutor;
};

#endif // SYSTEMINFOPROVIDER_HPP
