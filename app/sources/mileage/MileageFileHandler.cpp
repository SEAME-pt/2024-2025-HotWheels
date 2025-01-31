#include "MileageFileHandler.hpp"
#include <QDebug>
#include <QFile>
#include <QTextStream>

MileageFileHandler::MileageFileHandler(const QString &filePath)
    : filePath(filePath) {
  ensureFileExists();
}

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
