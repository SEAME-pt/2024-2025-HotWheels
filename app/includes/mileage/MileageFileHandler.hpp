#ifndef MILEAGEFILEHANDLER_HPP
#define MILEAGEFILEHANDLER_HPP

#include <QFile>
#include <QString>
#include "FileController.hpp"
#include "IMileageFileHandler.hpp"
#include <functional>

using FileOpenFunc = std::function<bool(QFile &, QIODevice::OpenMode)>;
using FileReadFunc = std::function<QString(QFile &)>;
using FileWriteFunc = std::function<bool(QFile &, const QString &)>;
using FileExistsFunc = std::function<bool(const QString &)>;

class MileageFileHandler : public IMileageFileHandler
{
public:
    explicit MileageFileHandler(const QString &filePath,
                                FileOpenFunc openFunc = FileController::open,
                                FileReadFunc readFunc = FileController::read,
                                FileWriteFunc writeFunc = FileController::write,
                                FileExistsFunc existsFunc = FileController::exists);

    ~MileageFileHandler() = default;

    double readMileage() const override;
    void writeMileage(double mileage) const override;

private:
    QString filePath;
    FileOpenFunc openFunc;
    FileReadFunc readFunc;
    FileWriteFunc writeFunc;
    FileExistsFunc existsFunc;

    void ensureFileExists() const;
};

#endif // MILEAGEFILEHANDLER_HPP
