#ifndef ISYSTEMCOMMANDEXECUTOR_HPP
#define ISYSTEMCOMMANDEXECUTOR_HPP

#include <QString>

class ISystemCommandExecutor
{
public:
    virtual ~ISystemCommandExecutor() = default;
    virtual QString executeCommand(const QString &command) const = 0;
    virtual QString readFile(const QString &filePath) const = 0;
};

#endif // ISYSTEMCOMMANDEXECUTOR_HPP
