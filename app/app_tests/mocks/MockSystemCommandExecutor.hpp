#ifndef MOCKSYSTEMCOMMANDEXECUTOR_HPP
#define MOCKSYSTEMCOMMANDEXECUTOR_HPP

#include "ISystemCommandExecutor.hpp"
#include <gmock/gmock.h>

class MockSystemCommandExecutor : public ISystemCommandExecutor
{
public:
    MOCK_METHOD(QString, executeCommand, (const QString &command), (const, override));
    MOCK_METHOD(QString, readFile, (const QString &filePath), (const, override));
};

#endif // MOCKSYSTEMCOMMANDEXECUTOR_HPP
