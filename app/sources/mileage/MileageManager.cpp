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
#include "MileageCalculator.hpp"
#include "MileageFileHandler.hpp"

/**
 * @brief Constructs a MileageManager object with the specified file path, calculator, and file handler.
 * @param filePath The path to the mileage file.
 * @param calculator The mileage calculator to use.
 * @param fileHandler The file handler to use.
 * @param parent The parent object.
 *
 * @details This constructor initializes the MileageManager object with the specified
 * file path, calculator, and file handler. It also sets the total mileage to 0.0.
 */
MileageManager::MileageManager(const QString &filePath,
							   IMileageCalculator *calculator,
							   IMileageFileHandler *fileHandler,
							   QObject *parent)
	: QObject(parent)
	, m_calculator(calculator ? calculator : new MileageCalculator())
	, m_fileHandler(fileHandler ? fileHandler : new MileageFileHandler(filePath))
	, m_ownCalculator(calculator == nullptr)
	, m_ownFileHandler(fileHandler == nullptr)
	, m_totalMileage(0.0)
{}

/**
 * @brief Destructs the MileageManager object.
 *
 * @details This destructor calls the shutdown method to stop the timers and
 * saves the mileage to the file. It also deletes the calculator and file handler
 * if they were created internally.
 */
MileageManager::~MileageManager()
{
	shutdown();

	// Only delete instances if they were created internally
	if (m_ownCalculator) {
		delete m_calculator;
	}
	if (m_ownFileHandler) {
		delete m_fileHandler;
	}
}

/**
 * @brief Initializes the MileageManager object.
 *
 * @details This method initializes the MileageManager object by reading the
 * mileage from the file and starting the update and persistence timers.
 */
void MileageManager::initialize()
{
	m_totalMileage = m_fileHandler->readMileage();

	connect(&m_updateTimer, &QTimer::timeout, this, &MileageManager::updateMileage);
	m_updateTimer.start(1000);

	connect(&m_persistenceTimer, &QTimer::timeout, this, &MileageManager::saveMileage);
	m_persistenceTimer.start(10000);
}

/**
 * @brief Shuts down the MileageManager object.
 *
 * @details This method stops the update and persistence timers and saves the
 * mileage to the file.
 */
void MileageManager::shutdown()
{
	saveMileage();
	m_updateTimer.stop();
	m_persistenceTimer.stop();
}

/**
 * @brief Updates the mileage.
 *
 * @details This method updates the mileage by calculating the distance traveled
 * since the last update and adding it to the total mileage.
 */
void MileageManager::updateMileage()
{
	double distance = m_calculator->calculateDistance();
	m_totalMileage += distance;
	emit mileageUpdated(m_totalMileage);
}

/**
 * @brief Saves the mileage to the file.
 *
 * @details This method saves the total mileage to the file using the file handler.
 */
void MileageManager::saveMileage()
{
	m_fileHandler->writeMileage(m_totalMileage);
}

/**
 * @brief Handles the speed updated signal.
 * @param speed The new speed value.
 *
 * @details This method handles the speed updated signal by adding the speed value
 * to the calculator.
 */
void MileageManager::onSpeedUpdated(float speed)
{
	m_calculator->addSpeed(speed);
}