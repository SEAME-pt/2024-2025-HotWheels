#ifndef SYSTEMCOMMANDEXECUTOR_HPP
#define SYSTEMCOMMANDEXECUTOR_HPP

#include <QFile>
#include <QProcess>
#include <QTextStream>
#include "ISystemCommandExecutor.hpp"

class SystemCommandExecutor : public ISystemCommandExecutor
{
public:
    QString executeCommand(const QString &command) const override;
    QString readFile(const QString &filePath) const override;
};

#endif // SYSTEMCOMMANDEXECUTOR_HPP
