#include "FileController.hpp"
#include <QDebug>
#include <QTextStream>

namespace FileController {

bool open(QFile &file, QIODevice::OpenMode mode)
{
    return file.open(mode);
}

QString read(QFile &file)
{
    QTextStream in(&file);
    return in.readLine();
}

bool write(QFile &file, const QString &data)
{
    QTextStream out(&file);
    out << data << Qt::endl;
    return true;
}

bool exists(const QString &path)
{
    return QFile::exists(path);
}

} // namespace FileController
