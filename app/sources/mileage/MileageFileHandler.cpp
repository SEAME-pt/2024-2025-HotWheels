/**
 * @file MileageFileHandler.cpp
 * @brief Implementation of the MileageFileHandler class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the MileageFileHandler class,
 * which is used to read and write the mileage to a file.
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
#include <QFile>
#include <QTextStream>

/**
 * @brief Construct a new MileageFileHandler object
 *
 * @param filePath The path of the mileage file to manage.
 * @details This constructor initializes the MileageFileHandler object with the
 * specified file path and ensures that the file exists.
 */
MileageFileHandler::MileageFileHandler(const QString &filePath)
    : filePath(filePath) {
  ensureFileExists();
}

/**
 * @brief Ensures that the mileage file exists.
 * If the file does not exist, it creates the file and initializes it with a
 * mileage of 0.0.
 * @details This function checks if the mileage file exists and creates it if it
 * does not exist.
 */
void MileageFileHandler::ensureFileExists() const {
  QFile file(filePath);
  if (!file.exists()) {
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
      QTextStream out(&file);
      out << "0.0\n"; // Initialize with a mileage of 0.0
      file.close();
      qDebug() << "Mileage file created at:" << filePath;
    } else {
      qWarning() << "Failed to create mileage file at:" << filePath;
    }
  }
}

/**
 * @brief This function reads the mileage value from the file and returns it.
 * @details Reads the mileage value from the file.
 * If the file cannot be opened or contains an invalid value, a default
 * mileage of 0.0 is returned.
 *
 * @return The mileage read from the file, or 0.0 if there was an issue.
 */
double MileageFileHandler::readMileage() const {
  QFile file(filePath);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    qWarning() << "Failed to open mileage file for reading:" << filePath;
    return 0.0; // Default mileage
  }

  QTextStream in(&file);
  double mileage = 0.0;
  if (!in.atEnd()) {
    QString line = in.readLine();
    bool ok = false;
    mileage = line.toDouble(&ok);
    if (!ok) {
      qWarning() << "Invalid mileage value in file. Defaulting to 0.";
      mileage = 0.0;
    }
  }

  file.close();
  return mileage;
}

/**
 * @brief This function writes the provided mileage value to the file.
 * @details Writes the provided mileage value to the file.
 * The value is saved with two decimal precision.
 *
 * @param mileage The mileage value to be written to the file.
 */
void MileageFileHandler::writeMileage(double mileage) const {
  QFile file(filePath);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    qWarning() << "Failed to open mileage file for writing:" << filePath;
    return;
  }

  QTextStream out(&file);
  out << QString::number(mileage, 'f', 2) << Qt::endl;
  file.close();
}
