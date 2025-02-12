/*!
 * @file ISystemCommandExecutor.hpp
 * @brief Definition of the ISystemCommandExecutor interface.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the ISystemCommandExecutor interface, which
 * is responsible for executing system commands and reading files.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef ISYSTEMCOMMANDEXECUTOR_HPP
#define ISYSTEMCOMMANDEXECUTOR_HPP

#include <QString>

/*!
 * @brief Interface for executing system commands and reading files.
 * @class ISystemCommandExecutor
 */
class ISystemCommandExecutor
{
public:
    virtual ~ISystemCommandExecutor() = default;
    virtual QString executeCommand(const QString &command) const = 0;
    virtual QString readFile(const QString &filePath) const = 0;
};

#endif // ISYSTEMCOMMANDEXECUTOR_HPP
