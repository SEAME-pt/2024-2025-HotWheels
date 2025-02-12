/*!
 * @file SystemCommandExecutor.hpp
 * @brief Definition of the SystemCommandExecutor class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the SystemCommandExecutor class, which
 * is responsible for executing system commands and reading files.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef SYSTEMCOMMANDEXECUTOR_HPP
#define SYSTEMCOMMANDEXECUTOR_HPP

#include <QFile>
#include <QProcess>
#include <QTextStream>
#include "ISystemCommandExecutor.hpp"

/*!
 * @brief Class that executes system commands and reads files.
 * @class SystemCommandExecutor
 */
class SystemCommandExecutor : public ISystemCommandExecutor
{
public:
	QString executeCommand(const QString &command) const override;
	QString readFile(const QString &filePath) const override;
};

#endif // SYSTEMCOMMANDEXECUTOR_HPP
