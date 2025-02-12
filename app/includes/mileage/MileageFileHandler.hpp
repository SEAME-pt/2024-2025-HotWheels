/*!
 * @file MileageFileHandler.hpp
 * @brief Definition of the MileageFileHandler class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the MileageFileHandler class,
 * which is responsible for managing the mileage file.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

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

/*!
 * @brief Class that manages the mileage file.
 * @class MileageFileHandler
 */
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
