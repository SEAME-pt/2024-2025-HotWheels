/**
 * @file MileageManager.cpp
 * @brief Implementation of the MileageManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the MileageManager class,
 * which is used to manage the mileage of the vehicle.
 * @note This class is used to manage the mileage of the vehicle.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the file path is valid and accessible.
 * @see MileageManager.hpp
 * @copyright Copyright (c) 2025
 */

#include "MileageManager.hpp"
#include <QDebug>

/**
 * @brief Construct a new MileageManager object
 *
 * @param filePath The path of the mileage file to manage.
 * @param parent The parent QObject.
 * @details This constructor initializes the MileageManager object with the
 * specified file path.
 */
MileageManager::MileageManager(const QString &filePath, QObject *parent)
		: QObject(parent), fileHandler(filePath), totalMileage(0.0) {}

MileageManager::~MileageManager() { shutdown(); }

/**
 * @brief Initialize the MileageManager.
 * @details This function initializes the MileageManager by loading the initial
 * mileage from the file and starting the update and persistence timers.
 */
void MileageManager::initialize() {
	// Load initial mileage from file
	totalMileage = fileHandler.readMileage();

	// Configure update timer (every 5 seconds)
	connect(&updateTimer, &QTimer::timeout, this, &MileageManager::updateMileage);
	updateTimer.start(1000);

	// Configure persistence timer (every 10 seconds)
	connect(&persistenceTimer, &QTimer::timeout, this,
					&MileageManager::saveMileage);
	persistenceTimer.start(10000);
}

/**
 * @brief Shutdown the MileageManager.
 * @details This function stops the update and persistence timers and saves the
 * mileage to the file.
 */
void MileageManager::shutdown() {
	saveMileage(); // Ensure mileage is saved on shutdown
	updateTimer.stop();
	persistenceTimer.stop();
}

/**
 * @brief Slot for handling speed updates.
 * @param speed The current speed of the vehicle.
 * @details This function is called when the speed of the vehicle is updated.
 */
void MileageManager::onSpeedUpdated(float speed) { calculator.addSpeed(speed); }

/**
 * @brief Update the mileage of the vehicle.
 * @details This function calculates the distance traveled by the vehicle and
 * updates the total mileage.
 */
void MileageManager::updateMileage() {
	// Calculate distance for the last interval
	// qDebug() << "Updating mileage";
	double distance = calculator.calculateDistance();
	totalMileage += distance;

	// Emit updated mileage
	// qDebug() << "Updating mileage" << totalMileage;
	emit mileageUpdated(totalMileage);
}

/**
 * @brief Save the mileage to the file.
 * @details This function saves the total mileage to the file.
 */
void MileageManager::saveMileage() { fileHandler.writeMileage(totalMileage); }
