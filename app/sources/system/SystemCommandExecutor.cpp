#include "SystemCommandExecutor.hpp"

QString SystemCommandExecutor::executeCommand(const QString &command) const
{
    QProcess process;
    process.start("sh", {"-c", command});
    process.waitForFinished();
    return process.readAllStandardOutput().trimmed();
}

QString SystemCommandExecutor::readFile(const QString &filePath) const
{
    QFile file(filePath);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        return in.readLine().trimmed();
    }
    return "";
}
