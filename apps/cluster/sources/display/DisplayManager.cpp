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
    if (!m_ui->speedLabel || !m_ui->timeLabel ||
			!m_ui->temperatureLabel || !m_ui->batteryLabel) {
		qDebug() << "Error: Labels not initialized in the UI form!";
		return;
	}

	m_ui->speedLimit50Label->hide();
	m_ui->speedLimit80Label->hide();

	// Set initial values for the labels
	m_ui->speedLabel->setText("0");
	m_ui->timeLabel->setText("--:--:--");
	m_ui->temperatureLabel->setText("N/A");
	m_ui->batteryLabel->setText("--%");
	m_ui->speedMetricsLabel->setText("KM/H");

	setupWifiDropdown();

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
		m_ui->directionParkLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionReverseLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionNeutralLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionDriveLabel->setStyleSheet("color: green; background: transparent;");

		m_ui->directionParkLabel->update();
		m_ui->directionReverseLabel->update();
		m_ui->directionNeutralLabel->update();
		m_ui->directionDriveLabel->update();
		break;
	case CarDirection::Reverse:
		//directionText = "R";
		m_ui->directionParkLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionReverseLabel->setStyleSheet("color: lightgreen; background: transparent;");
		m_ui->directionNeutralLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionDriveLabel->setStyleSheet("color: white; background: transparent;");

		m_ui->directionParkLabel->update();
		m_ui->directionReverseLabel->update();
		m_ui->directionNeutralLabel->update();
		m_ui->directionDriveLabel->update();
		break;
	case CarDirection::Stop:
	default:
		//directionText = "D";
		m_ui->directionParkLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionReverseLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionNeutralLabel->setStyleSheet("color: white; background: transparent;");
		m_ui->directionDriveLabel->setStyleSheet("color: lightgreen; background: transparent;");

		m_ui->directionParkLabel->update();
		m_ui->directionReverseLabel->update();
		m_ui->directionNeutralLabel->update();
		m_ui->directionDriveLabel->update();
		break;
	}
}

/*!
 * @brief Updates the system time on the display.
 * @details This function updates the date, time, and weekday labels based on
 * the current system time.
 * @param currentMonth The current month.
 * @param currentTime The current time.
 * @param currentDay The current day of the week.
 */
void DisplayManager::updateSystemTime(const QString &currentMonth,
																			const QString &currentTime,
                                                                            const QString &currentDay) {
	m_ui->dateLabel->setText(currentMonth + " " + currentDay);
	m_ui->timeLabel->setText(currentTime);
}

QString getWifiSSID() {
    // Linux: read active SSID using shell command
    QProcess proc;
    proc.start("iwgetid -r");  // Gets current SSID
    proc.waitForFinished();
    QString output = proc.readAllStandardOutput().trimmed();
    return output.isEmpty() ? "Not connected" : output;
}

QString getLocalIPAddress() {
    const QList<QHostAddress> &addresses = QNetworkInterface::allAddresses();
    for (const QHostAddress &address : addresses) {
        if (address.protocol() == QAbstractSocket::IPv4Protocol && !address.isLoopback())
            return address.toString();
    }
    return "No IP";
}

void DisplayManager::setupWifiDropdown() {
    connect(m_ui->wifiToggleButton, &QToolButton::clicked, this, [=]() {
        QMenu* wifiMenu = new QMenu(m_ui->wifiToggleButton);

        // Fetch current data
        QString ssidText = getWifiSSID();
        QString ipText = getLocalIPAddress();

        wifiMenu->addAction("Connected to: " + ssidText);
        wifiMenu->addAction("IP Address: " + ipText);

        wifiMenu->setStyleSheet(
            "QMenu {"
            " background-color: rgba(30, 30, 30, 0.9);"
            " color: white;"
            " border: 1px solid rgba(255, 255, 255, 0.2);"
            " border-radius: 6px;"
            " padding: 6px;"
            " }"
        );

        QPoint pos = m_ui->wifiToggleButton->mapToGlobal(QPoint(0, m_ui->wifiToggleButton->height()));
        wifiMenu->exec(pos);
    });
}

/*!
 * @brief Updates the temperature on the display.
 * @details This function updates the temperature label based on the current
 * temperature.
 * @param temperature The current temperature.
 */
void DisplayManager::updateTemperature(const QString &temperature) {
	m_ui->temperatureLabel->setText(temperature);
}

/*!
 * @brief Updates the battery percentage on the display.
 * @details This function updates the battery percentage label and low battery
 * warning based on the current battery percentage.
 * @param batteryPercentage The current battery percentage.
 */
void DisplayManager::updateBatteryPercentage(float batteryPercentage) {
	m_ui->batteryLabel->setText(QString::number(static_cast<int>(batteryPercentage)) + "%");
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
		m_ui->laneKeepingAssistLabel->show();
		m_ui->laneDepartureWarningLabel->show();
		// Stop blinking if active
		if (m_blinkTimer) {
			m_blinkTimer->stop();
			m_blinkTimer->deleteLater();
			m_blinkTimer = nullptr;
		}
		break;
	case DrivingMode::Automatic:
		modeText = "Automatic";
		m_ui->laneKeepingAssistLabel->hide();
		if (!m_blinkTimer) {
			m_blinkTimer = new QTimer(this);
			connect(m_blinkTimer, &QTimer::timeout, this, [=]() {
				bool currentlyVisible = m_ui->laneDepartureWarningLabel->isVisible();
				m_ui->laneDepartureWarningLabel->setVisible(!currentlyVisible);
			});
			m_blinkTimer->start(150);  // Blink every 150ms
		}
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

void DisplayManager::updateSpeedLimitLabels(int speed) {
	if (speed == 50) {
		m_ui->speedLimit80Label->hide();
		m_ui->speedLimit50Label->show();
		QTimer::singleShot(3000, m_ui->speedLimit50Label, &QWidget::hide);
	} else if (speed == 80) {
		m_ui->speedLimit50Label->hide();
		m_ui->speedLimit80Label->show();
		QTimer::singleShot(3000, m_ui->speedLimit80Label, &QWidget::hide);
	}
}

void DisplayManager::displayInferenceImage(const QImage &image) {
	if (!m_ui->inferenceLabel)
		return;

	QPixmap original = QPixmap::fromImage(image);
	QPixmap rounded(original.size());
	rounded.fill(Qt::transparent);

	QPainter painter(&rounded);
	painter.setRenderHint(QPainter::Antialiasing, true);

	QPainterPath path;
	path.addRoundedRect(original.rect(), 34, 34);
	painter.setClipPath(path);
	painter.drawPixmap(0, 0, original);

	m_ui->inferenceLabel->setPixmap(rounded);
}
