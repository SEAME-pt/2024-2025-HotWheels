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

/**
 * @brief Constructs a MileageFileHandler object with the specified file path and functions.
 * @param filePath The path to the mileage file.
 * @param openFunc The function to open a file.
 * @param readFunc The function to read from a file.
 * @param writeFunc The function to write to a file.
 * @param existsFunc The function to check if a file exists.
 *
 * @details This constructor initializes the MileageFileHandler object with the specified
 * file path and functions. It also calls ensureFileExists() to create the file if it
 * does not exist.
 */
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

/**
 * @brief Checks if the file exists and creates it if it does not.
 *
 * @details This method checks if the file exists by calling the existsFunc
 * function. If the file does not exist, it is created by calling the openFunc
 * function with the QIODevice::WriteOnly and QIODevice::Text flags. The default
 * value of the mileage is written to the file by calling the writeFunc function.
 * If the file is created successfully, a message is logged to the console
 * indicating that the file was created.
 *
 * @warning If the file cannot be created, a warning message is logged to the
 * console.
 */
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

/**
 * @brief Reads the mileage from the file.
 * @return The mileage value read from the file or 0.0 if the file is invalid.
 *
 * @details This method reads the mileage value from the file by calling the
 * readFunc function. If the file cannot be opened for reading, a warning message
 * is logged to the console. If the mileage value is invalid, a warning message is
 * logged to the console and the default value of 0.0 is returned.
 */
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
/**
 * @brief Writes the mileage to the file.
 * @param mileage The mileage value to write.
 *
 * @details This method writes the mileage value to the file by calling the
 * writeFunc function. If the file cannot be opened for writing, a warning message
 * is logged to the console.
 */
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
