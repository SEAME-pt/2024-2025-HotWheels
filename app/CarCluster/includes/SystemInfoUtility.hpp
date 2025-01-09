#ifndef SYSTEMINFOUTILITY_HPP
#define SYSTEMINFOUTILITY_HPP

#include <QString>

class SystemInfoUtility
{
public:
    static QString getWifiStatus();
    static QString getTemperature();
};

#endif // SYSTEMINFOUTILITY_HPP
