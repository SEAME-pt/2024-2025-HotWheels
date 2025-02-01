/**
 * @file DisplayManager.hpp
 * @brief Definition of the DisplayManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the DisplayManager class, which
 * is responsible for managing the display of the car manager.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef DISPLAYMANAGER_HPP
#define DISPLAYMANAGER_HPP

#include "enums.hpp"
#include "ui_CarManager.h"
#include <QObject>
#include <QString>

/**
 * @brief Class that manages the display of the car manager.
 * @class DisplayManager inherits from QObject
 */
class DisplayManager : public QObject {
	Q_OBJECT

public:
	explicit DisplayManager(Ui::CarManager *ui, QObject *parent = nullptr);

public slots:
	void updateCanBusData(float speed, int rpm);
	void updateEngineData(CarDirection direction, int steeringAngle);
	void updateSystemTime(const QString &currentDate, const QString &currentTime,
												const QString &currentDay);
	void updateWifiStatus(const QString &status, const QString &wifiName);
	void updateTemperature(const QString &temperature);
	void updateBatteryPercentage(float batteryPercentage);
	void updateIpAddress(const QString &ipAddress);
	void updateMileage(double mileage);
	void updateDrivingMode(DrivingMode newMode);
	void updateClusterTheme(ClusterTheme newTheme);
	void updateClusterMetrics(ClusterMetrics newMetrics);

signals:
	/** @brief Signal emitted when the driving mode is toggled. */
	void drivingModeToggled();
	/** @brief Signal emitted when the cluster theme is toggled. */
	void clusterThemeToggled();
	/** @brief Signal emitted when the cluster metrics are toggled. */
	void clusterMetricsToggled();

private:
	/** @brief Pointer to the UI object. */
	Ui::CarManager *m_ui;
};

#endif // DISPLAYMANAGER_HPP
