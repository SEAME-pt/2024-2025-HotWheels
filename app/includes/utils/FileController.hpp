#ifndef FILECONTROLLER_HPP
#define FILECONTROLLER_HPP

#include <QFile>
#include <QString>

namespace FileController {
bool open(QFile &file, QIODevice::OpenMode mode);
QString read(QFile &file);
bool write(QFile &file, const QString &data);
bool exists(const QString &path);
} // namespace FileController

#endif // FILECONTROLLER_HPP
