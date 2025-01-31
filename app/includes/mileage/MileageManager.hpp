#ifndef MILEAGEMANAGER_HPP
#define MILEAGEMANAGER_HPP

#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"
#include <QObject>
#include <QTimer>

class MileageManager : public QObject {
  Q_OBJECT

public:
  /**
   * Constructor for the MileageManager class.
   * Initializes the class with a file path for mileage data and sets up the
   * file handler.
   *
   * @param filePath The path of the file to store and load mileage data.
   * @param parent The parent QObject.
   */
  explicit MileageManager(const QString &filePath, QObject *parent = nullptr);

  /**
   * Destructor for the MileageManager class.
   * Ensures that mileage is saved before shutdown.
   */
  ~MileageManager();

  /**
   * Initializes the MileageManager:
   * - Loads the initial mileage from the file.
   * - Configures update timer (every 1 second).
   * - Configures persistence timer (every 10 seconds).
   */
  void initialize();

  /**
   * Shuts down the MileageManager:
   * - Saves the current mileage to the file.
   * - Stops the update and persistence timers.
   */
  void shutdown();

public slots:
  /**
   * Slot to receive speed updates.
   * It forwards the received speed data to the MileageCalculator.
   *
   * @param speed The current speed of the vehicle.
   */
  void onSpeedUpdated(float speed);

  /**
   * Slot to periodically update the mileage.
   * It calculates the total distance for the last interval and adds it to the
   * total mileage.
   */
  void updateMileage();

  /**
   * Slot to save the current mileage to the file.
   * This is called periodically to ensure the data is persisted.
   */
  void saveMileage();

signals:
  /**
   * Signal emitted when the mileage is updated.
   *
   * @param mileage The current total mileage.
   */
  void mileageUpdated(double mileage);

private:
  MileageCalculator calculator; /**< A calculator that computes the distance
                                   based on speed and time intervals. */
  MileageFileHandler
      fileHandler; /**< A file handler to read and write mileage data. */
  QTimer
      updateTimer; /**< A timer that triggers mileage updates every 1 second. */
  QTimer persistenceTimer; /**< A timer that triggers saving mileage every 10
                              seconds. */
  double totalMileage;     /**< The current total mileage of the vehicle. */
};

#endif // MILEAGEMANAGER_HPP
