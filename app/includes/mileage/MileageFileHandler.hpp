/**
 * @file MileageFileHandler.hpp
 * @brief 
 * @version 0.1
 * @date 2025-01-31
 * @details
 * @note 
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

class MileageFileHandler {
public:
  /**
   * Constructor for the MileageFileHandler class.
   * It initializes the class with the specified file path and ensures that the
   * file exists.
   *
   * @param filePath The path of the mileage file to be managed.
   */
  explicit MileageFileHandler(const QString &filePath);

  /**
   * Destructor for the MileageFileHandler class.
   * The destructor is implicitly defaulted as no manual resource management is
   * required.
   */
  ~MileageFileHandler() = default;

  /**
   * Reads the mileage value from the file.
   * If the file cannot be opened or contains an invalid value, a default
   * mileage of 0.0 is returned.
   *
   * @return The mileage read from the file, or 0.0 if there was an issue.
   */
  double readMileage() const;

  /**
   * Writes the provided mileage value to the file.
   * The value is saved with two decimal precision.
   *
   * @param mileage The mileage value to be written to the file.
   */
  void writeMileage(double mileage) const;

  /**
   * Ensures that the mileage file exists.
   * If the file does not exist, it creates the file and initializes it with a
   * mileage of 0.0.
   */
  void ensureFileExists() const;

private:
  QString filePath; /**< The path to the mileage file to be managed. */
};

#endif // MILEAGEFILEHANDLER_HPP
