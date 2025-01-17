#ifndef SYSTEMINFOUTILITY_HPP
#define SYSTEMINFOUTILITY_HPP

#include <QString>

class SystemInfoUtility
{
public:
    enum class InfoType { DesBegin, DesEnd, CreBegin, CreEnd };

    static QString getWifiStatus();
    static QString getTemperature();

    static void printClassInfo(QString className, InfoType type);
};

#endif // SYSTEMINFOUTILITY_HPP
