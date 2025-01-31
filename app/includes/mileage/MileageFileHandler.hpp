/**
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

#include <QString>

/**
 * @brief Class that manages the mileage file.
 * @class MileageFileHandler
 */
class MileageFileHandler {
public:
  explicit MileageFileHandler(const QString &filePath);
  ~MileageFileHandler() = default;
  double readMileage() const;
  void writeMileage(double mileage) const;
  void ensureFileExists() const;

private:
  /** @brief The path to the mileage file to be managed. */
  QString filePath;
};

#endif // MILEAGEFILEHANDLER_HPP
