/*!
 * @file ISystemInfoProvider.hpp
 * @brief Definition of the ISystemInfoProvider interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the ISystemInfoProvider interface, which
 * is responsible for providing system information to the display manager.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ISYSTEMINFOPROVIDER_HPP
#define ISYSTEMINFOPROVIDER_HPP

#include <QString>

/*!
 * @brief Interface for providing system information to the display manager.
 * @class ISystemInfoProvider
 */
class ISystemInfoProvider
{
public:
    virtual ~ISystemInfoProvider() = default;
    virtual QString getWifiStatus(QString &wifiName) const = 0;
    virtual QString getTemperature() const = 0;
    virtual QString getIpAddress() const = 0;
};

#endif // ISYSTEMINFOPROVIDER_HPP
