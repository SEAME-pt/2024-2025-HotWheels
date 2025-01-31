/**
 * @file MileageManager.hpp
 * @brief Definition of the MileageManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the MileageManager class, which
 * is responsible for managing the mileage of a vehicle.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MILEAGEMANAGER_HPP
#define MILEAGEMANAGER_HPP

#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"
#include <QObject>
#include <QTimer>

/**
 * @brief Class that manages the mileage of a vehicle.
 * @class MileageManager inherits from QObject
 */
class MileageManager : public QObject {
  Q_OBJECT

public:
  explicit MileageManager(const QString &filePath, QObject *parent = nullptr);
  ~MileageManager();
  void initialize();
  void shutdown();

public slots:

  void onSpeedUpdated(float speed);
  void updateMileage();
  void saveMileage();

signals:
  /**
   * @brief Signal emitted when the mileage is updated.
   * @param mileage The new mileage value.
   */
  void mileageUpdated(double mileage);

private:
  /** @brief The calculator that computes the distance based on speed and time
   * intervals. */
  MileageCalculator calculator; 
  /** @brief The file handler to read and write mileage data. */
  MileageFileHandler fileHandler;
  /** @brief A timer that triggers mileage updates every 1 second. */
  QTimer updateTimer; 
  /** @brief A timer that triggers saving mileage every 10 seconds. */
  QTimer persistenceTimer;
  /** @brief The current total mileage of the vehicle. */  
  double totalMileage;
};

#endif // MILEAGEMANAGER_HPP
