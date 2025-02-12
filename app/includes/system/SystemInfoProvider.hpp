/*!
 * @file SystemInfoProvider.hpp
 * @brief Definition of the SystemInfoProvider class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the SystemInfoProvider class, which
 * is responsible for providing system information to the display manager.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SYSTEMINFOPROVIDER_HPP
#define SYSTEMINFOPROVIDER_HPP

#include "ISystemCommandExecutor.hpp"
#include "ISystemInfoProvider.hpp"

/*!
 * @brief Class that provides system information to the display manager.
 * @class SystemInfoProvider
 */
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
