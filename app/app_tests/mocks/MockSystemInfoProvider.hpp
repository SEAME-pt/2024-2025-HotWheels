#ifndef MOCKSYSTEMINFOPROVIDER_HPP
#define MOCKSYSTEMINFOPROVIDER_HPP

#include "ISystemInfoProvider.hpp"
#include <gmock/gmock.h>

class MockSystemInfoProvider : public ISystemInfoProvider
{
public:
    MOCK_METHOD(QString, getWifiStatus, (QString & wifiName), (const, override));
    MOCK_METHOD(QString, getTemperature, (), (const, override));
    MOCK_METHOD(QString, getIpAddress, (), (const, override));
};

#endif // MOCKSYSTEMINFOPROVIDER_HPP
