/*!
 * @file MileageFileHandler.cpp
 * @brief Implementation of the MileageFileHandler class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the MileageFileHandler
 * class, which is used to read and write the mileage to a file.
 * @note This class is used to read and write the mileage to a file.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the file path is valid and accessible.
 * @see MileageFileHandler.hpp
 * @copyright Copyright (c) 2025
 */

#include "MileageFileHandler.hpp"
#include <QDebug>

MileageFileHandler::MileageFileHandler(const QString &filePath,
                                       FileOpenFunc openFunc,
                                       FileReadFunc readFunc,
                                       FileWriteFunc writeFunc,
                                       FileExistsFunc existsFunc)
    : filePath(filePath)
    , openFunc(openFunc)
    , readFunc(readFunc)
    , writeFunc(writeFunc)
    , existsFunc(existsFunc)
{
    ensureFileExists();
}

void MileageFileHandler::ensureFileExists() const
{
    if (!existsFunc(filePath)) {
        QFile file(filePath);
        if (openFunc(file, QIODevice::WriteOnly | QIODevice::Text)) {
            if (!writeFunc(file, "0.0")) {
                qWarning() << "Failed to initialize mileage file with default value.";
            }
            file.close();
            qDebug() << "Mileage file created at:" << filePath;
        } else {
            qWarning() << "Failed to create mileage file at:" << filePath;
        }
    }
}

double MileageFileHandler::readMileage() const
{
    QFile file(filePath);
    if (!openFunc(file, QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open mileage file for reading:" << filePath;
        return 0.0;
    }

    QString content = readFunc(file);
    file.close();

    bool ok = false;
    double mileage = content.toDouble(&ok);
    if (!ok) {
        qWarning() << "Invalid mileage value in file. Defaulting to 0.";
        return 0.0;
    }
    return mileage;
}

void MileageFileHandler::writeMileage(double mileage) const
{
    QFile file(filePath);
    if (!openFunc(file, QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open mileage file for writing:" << filePath;
        return;
    }

    bool success = writeFunc(file, QString::number(mileage, 'f', 2));
    if (!success) {
        qWarning() << "Failed to write mileage data.";
    }
    file.close();
}
