/*!
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

#include <QObject>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>
#include <QUrl>
#include "IMileageCalculator.hpp"
#include "IMileageFileHandler.hpp"

/*!
 * @brief Class that manages the mileage of a vehicle.
 * @class MileageManager inherits from QObject
 */
class MileageManager : public QObject {
  Q_OBJECT

public:
	explicit MileageManager(const QString &filePath,
							IMileageCalculator *calculator = nullptr,
							IMileageFileHandler *fileHandler = nullptr,
							QObject *parent = nullptr);
	~MileageManager();
	void initialize();
	void shutdown();

	double getTotalMileage() const { return this->m_totalMileage; };

public slots:
	void onSpeedUpdated(float speed);
	void updateMileage();
	void saveMileage();

signals:
	void mileageUpdated(double mileage);

private:
	IMileageCalculator *m_calculator;
	IMileageFileHandler *m_fileHandler;
	bool m_ownCalculator;
	bool m_ownFileHandler;
	QTimer m_updateTimer;
	QTimer m_persistenceTimer;
	double m_totalMileage;
};

#endif // MILEAGEMANAGER_HPP
