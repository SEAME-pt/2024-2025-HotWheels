/*!
 * @file SystemCommandExecutor.cpp
 * @brief Implementation of the SystemCommandExecutor class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the SystemCommandExecutor
 * class, which is used to execute system commands and read files.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "SystemCommandExecutor.hpp"

/*!
 * @brief Executes a system command.
 * @param command The command to execute as a QString.
 * @return The standard output of the executed command, trimmed of any leading
 * or trailing whitespace.
 */

QString SystemCommandExecutor::executeCommand(const QString &command) const
{
    QProcess process;
    process.start("sh", {"-c", command});
    process.waitForFinished();
    return process.readAllStandardOutput().trimmed();
}

/*!
 * @brief Reads a file and returns its contents.
 * @param filePath The path to the file to read as a QString.
 * @return The contents of the file, trimmed of any leading or trailing
 * whitespace, or an empty string if the file could not be opened.
 */
QString SystemCommandExecutor::readFile(const QString &filePath) const
{
    QFile file(filePath);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        return in.readLine().trimmed();
    }
    return "";
}
