/*!
 * @file DisplayManager.cpp
 * @brief Implementation of the DisplayManager class for handling the display of
 * the cluster.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @details This file contains the implementation of the DisplayManager class,
 * which is responsible for handling the display of the cluster.
 * @note This class is used to manage the display of the cluster.
 * @warning Ensure that the display labels are properly initialized in the UI
 * form.
 * @see DisplayManager.hpp
 * @copyright Copyright (c) 2025
 */

#include "DisplayManager.hpp"
#include <QDebug>
#include <QPushButton>

/*!
 * @brief Construct a new DisplayManager object.
 * @param ui The UI form for the cluster.
 * @param parent The parent QObject.
 * @details This constructor initializes the DisplayManager object with the
 * specified UI form.
 */
DisplayManager::DisplayManager(Ui::CarManager *ui, QObject *parent)
		: QObject(parent), m_ui(ui) {
	// Ensure the labels are initialized
    if (!m_ui->speedLabel || !m_ui->rpmLabel ||
			!m_ui->timeLabel || !m_ui->wifiLabel || !m_ui->temperatureLabel ||
			!m_ui->batteryLabel) {
		qDebug() << "Error: Labels not initialized in the UI form!";
		return;
	}

	// Set initial values for the labels
	m_ui->speedLabel->setText("0");
	m_ui->rpmLabel->setText("0.00");
	m_ui->timeLabel->setText("--:--:--");
	m_ui->wifiLabel->setText("📶 Disconnected");
	m_ui->temperatureLabel->setText("🌡️ N/A");
	m_ui->batteryLabel->setText("--% 🔋");
	m_ui->speedMetricsLabel->setText("KM/H");
	m_ui->leftBlinkerLabel->setVisible(false);
	m_ui->rightBlinkerLabel->setVisible(false);

	// Directly connect button clicks to signals
	connect(m_ui->toggleDrivingModeButton, &QPushButton::clicked, this,
					&DisplayManager::drivingModeToggled);
	connect(m_ui->toggleMetricsButton, &QPushButton::clicked, this,
					&DisplayManager::clusterMetricsToggled);
}

/*!
 * @brief Updates the CAN bus data on the display.
 * @details This function updates the speed and RPM labels based on the CAN bus
 * data.
 * @param speed The current speed of the car.
 * @param rpm The current RPM of the car.
 */
void DisplayManager::updateCanBusData(float speed, int rpm) {
	m_ui->speedLabel->setText(QString::number(static_cast<int>(speed)));
	m_ui->rpmLabel->setText(
			QString::number(static_cast<double>(rpm) / 1000, 'f', 2));
}

/*!
 * @brief Updates the engine data on the display.
 * @details This function updates the direction label and blinker visibility
 * based on the engine data.
 * @param direction The current direction of the car.
 * @param steeringAngle The current steering angle of the car.
 */
void DisplayManager::updateEngineData(CarDirection direction,
																			int steeringAngle) {
	//QString directionText;
	switch (direction) {
	case CarDirection::Drive:
		//directionText = "D";
		m_ui->directionDriveLabel->setStyleSheet("color: blue;");
		break;
	case CarDirection::Reverse:
		//directionText = "R";
		m_ui->directionReverseLabel->setStyleSheet("color: blue;");
		break;
	case CarDirection::Stop:
	default:
		//directionText = "D";
		m_ui->directionDriveLabel->setStyleSheet("color: blue;");
		break;
	}

	//m_ui->directionLabel->setText(directionText);
	if (steeringAngle > 0) {
		m_ui->leftBlinkerLabel->setVisible(false);
		m_ui->rightBlinkerLabel->setVisible(true);
	} else if (steeringAngle < 0) {
		m_ui->rightBlinkerLabel->setVisible(false);
		m_ui->leftBlinkerLabel->setVisible(true);
	} else {
		m_ui->leftBlinkerLabel->setVisible(false);
		m_ui->rightBlinkerLabel->setVisible(false);
	}
}

/*!
 * @brief Updates the system time on the display.
 * @details This function updates the date, time, and weekday labels based on
 * the current system time.
 * @param currentDate The current date.
 * @param currentTime The current time.
 * @param currentDay The current day of the week.
 */
void DisplayManager::updateSystemTime(const QString &currentDate,
																			const QString &currentTime,
                                                                            const QString &currentDay) {
	m_ui->dateLabel->setText(currentDate);
	m_ui->timeLabel->setText(currentTime);
}

/*!
 * @brief Updates the WiFi status on the display.
 * @details This function updates the WiFi status label based on the current
 * WiFi status and name.
 * @param status The current WiFi status.
 * @param wifiName The name of the connected WiFi network.
 */
void DisplayManager::updateWifiStatus(const QString &status,
																			const QString &wifiName) {
	QString wifiDisplay = status;
	if (!wifiName.isEmpty()) {
		wifiDisplay += " (" + wifiName + ")";
	}
	m_ui->wifiLabel->setText("📶 " + wifiName);
}

/*!
 * @brief Updates the temperature on the display.
 * @details This function updates the temperature label based on the current
 * temperature.
 * @param temperature The current temperature.
 */
void DisplayManager::updateTemperature(const QString &temperature) {
	m_ui->temperatureLabel->setText("🌡️ " + temperature);
}

/*!
 * @brief Updates the battery percentage on the display.
 * @details This function updates the battery percentage label and low battery
 * warning based on the current battery percentage.
 * @param batteryPercentage The current battery percentage.
 */
void DisplayManager::updateBatteryPercentage(float batteryPercentage) {
	m_ui->batteryLabel->setText(QString::number(batteryPercentage, 'f', 1) +
															"% " + (batteryPercentage > 20.0 ? "🔋" : "🪫"));
}

/*!
 * @brief Updates the mileage on the display.
 * @details This function updates the mileage label based on the current
 * mileage.
 * @param mileage The current mileage.
 */
void DisplayManager::updateMileage(double mileage) {
	m_ui->mileageLabel->setText(QString::number(static_cast<int>(mileage)) +
															" m");
}

/*!
 * @brief Updates the IP address on the display.
 * @details This function updates the IP address label based on the current IP
 * address.
 * @param ipAddress The current IP address.
 */
void DisplayManager::updateIpAddress(const QString &ipAddress) {
	m_ui->ipAddressLabel->setText(ipAddress);
}

/*!
 * @brief Updates the driving mode on the display.
 * @details This function updates the driving mode label based on the current
 * driving mode.
 * @param newMode The new driving mode.
 */
void DisplayManager::updateDrivingMode(DrivingMode newMode) {
	QString modeText;
	switch (newMode) {
	case DrivingMode::Manual:
		modeText = "Manual";
		break;
	case DrivingMode::Automatic:
		modeText = "Automatic";
		break;
	}
	m_ui->drivingModeLabel->setText(modeText);
}

/*!
 * @brief Updates the cluster theme on the display.
 * @details This function updates the cluster theme label based on the current
 * cluster theme.
 * @param newTheme The new cluster theme.
 */
void DisplayManager::updateClusterTheme(ClusterTheme newTheme) {
	QString themeText;
	switch (newTheme) {
	case ClusterTheme::Dark:
		themeText = "Dark";
		break;
	case ClusterTheme::Light:
		themeText = "Light";
		break;
	}
}

/*!
 * @brief Updates the cluster metrics on the display.
 * @details This function updates the cluster metrics label and speed metrics
 * label based on the current cluster metrics.
 * @param newMetrics The new cluster metrics.
 */
void DisplayManager::updateClusterMetrics(ClusterMetrics newMetrics) {
	QString metricsText;
	switch (newMetrics) {
	case ClusterMetrics::Kilometers:
		metricsText = "km/h";
		break;
	case ClusterMetrics::Miles:
		metricsText = "mph";
		break;
	}

	m_ui->speedMetricsLabel->setText(metricsText.toUpper());
}
