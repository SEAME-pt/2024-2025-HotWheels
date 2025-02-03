#ifndef MOCKFILECONTROLLER_HPP
#define MOCKFILECONTROLLER_HPP

#include <QFile>
#include <QString>
#include <gmock/gmock.h>

class MockFileController
{
public:
    MOCK_METHOD(bool, open, (QFile &, QIODevice::OpenMode), ());
    MOCK_METHOD(QString, read, (QFile &), ());
    MOCK_METHOD(bool, write, (QFile &, const QString &), ());
    MOCK_METHOD(bool, exists, (const QString &), ());
};

#endif // MOCKFILECONTROLLER_HPP
