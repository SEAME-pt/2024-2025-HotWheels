/*!
 * @file DataManager.cpp
 * @brief Implementation of the DataManager class for handling various data
 * types.
 * @version 0.1
 * @date 2025-01-31
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @details This file contains the implementation of the DataManager class,
 * which is responsible for handling different types of data such as CAN data,
 * system data, battery data, and driving mode.
 * @note See DataManager.hpp for the class definition.
 * @warning Ensure proper data validation before processing.
 * @see DataManager.hpp
 * @copyright Copyright (c) 2025
 */

#include "DataManager.hpp"
#include <QDebug>

/*!
 * @brief Construct a new DataManager::DataManager object
 * @param parent The parent QObject.
 * @details This constructor initializes the DataManager object and the three
 * managers: VehicleDataManager, SystemDataManager, and ClusterSettingsManager.
 */
DataManager::DataManager(QObject *parent)
	: QObject(parent)
{
	// Initialize the three managers
	m_vehicleDataManager = new VehicleDataManager(this);
	m_systemDataManager = new SystemDataManager(this);
	m_clusterSettingsManager = new ClusterSettingsManager(this);

	// Connect signals from VehicleDataManager
	connect(m_vehicleDataManager,
			&VehicleDataManager::canDataProcessed,
			this,
			&DataManager::canDataProcessed);
	connect(m_vehicleDataManager,
			&VehicleDataManager::engineDataProcessed,
			this,
			&DataManager::engineDataProcessed);
	connect(m_vehicleDataManager,
			&VehicleDataManager::mileageUpdated,
			this,
			&DataManager::mileageUpdated);

	// Connect signals from SystemDataManager
	connect(m_systemDataManager,
			&SystemDataManager::systemTimeUpdated,
			this,
			&DataManager::systemTimeUpdated);
	connect(m_systemDataManager,
			&SystemDataManager::systemWifiUpdated,
			this,
			&DataManager::systemWifiUpdated);
	connect(m_systemDataManager,
			&SystemDataManager::systemTemperatureUpdated,
			this,
			&DataManager::systemTemperatureUpdated);
	connect(m_systemDataManager,
			&SystemDataManager::batteryPercentageUpdated,
			this,
			&DataManager::batteryPercentageUpdated);

	// Connect signals from ClusterSettingsManager
	connect(m_clusterSettingsManager,
			&ClusterSettingsManager::drivingModeUpdated,
			this,
			&DataManager::drivingModeUpdated);
	connect(m_clusterSettingsManager,
			&ClusterSettingsManager::clusterThemeUpdated,
			this,
			&DataManager::clusterThemeUpdated);
	connect(m_clusterSettingsManager,
			&ClusterSettingsManager::clusterMetricsUpdated,
			this,
			&DataManager::clusterMetricsUpdated);
}

/*!
 * @brief Destroy the DataManager::DataManager object
 * @details This destructor cleans up the resources used by the DataManager.
 */
DataManager::~DataManager()
{
	delete m_vehicleDataManager;
	delete m_systemDataManager;
	delete m_clusterSettingsManager;
}

/*!
 * @brief Handle CAN data.
 * @param frameID The frame ID of the CAN message.
 * @param data The data of the CAN message.
 * @details This function processes the CAN data by forwarding it to the
 * VehicleDataManager.
 */
void DataManager::handleRpmData(int rawRpm)
{
	m_vehicleDataManager->handleRpmData(rawRpm);
}

/*!
 * @brief Handle Speed data.
	* @param rawSpeed The raw speed data.
	* @details This function processes the speed data by forwarding it to the
	* VehicleDataManager.
	*/
void DataManager::handleSpeedData(float rawSpeed)
{
	m_vehicleDataManager->handleSpeedData(rawSpeed);
	// qDebug() << "Speed updated";
}

/*!
 * @brief Handle Steering data.
 * @param rawAngle The raw angle data.
 * @details This function processes the steering data by forwarding it to the
 * VehicleDataManager.
 */
void DataManager::handleSteeringData(int rawAngle)
{
	m_vehicleDataManager->handleSteeringData(rawAngle);
}

/*!
 * @brief Handle Direction data.
 * @param rawDirection The raw direction data.
 * @details This function processes the direction data by forwarding it to the
 * VehicleDataManager.
 */
void DataManager::handleDirectionData(CarDirection rawDirection)
{
	m_vehicleDataManager->handleDirectionData(rawDirection);
}

/*!
 * @brief Handle Engine data.
 * @param engineStatus The engine status.
 * @details This function processes the engine data by forwarding it to the
 * VehicleDataManager.
 */
void DataManager::handleMileageUpdate(double mileage)
{
	m_vehicleDataManager->handleMileageUpdate(mileage);
}

/*!
 * @brief Handle Time data.
 * @param currentDate The current date.
 * @param currentTime The current time.
 * @param currentDay The current day.
 * @details This function processes the time data by forwarding it to the
 * SystemDataManager.
 */
void DataManager::handleTimeData(const QString &currentDate,
								 const QString &currentTime,
								 const QString &currentDay)
{
	m_systemDataManager->handleTimeData(currentDate, currentTime, currentDay);
}

/*!
 * @brief Handle WiFi data.
 * @param status The WiFi status.
 * @param wifiName The WiFi name.
 * @details This function processes the WiFi data by forwarding it to the
 * SystemDataManager.
 */
void DataManager::handleWifiData(const QString &status, const QString &wifiName)
{
	m_systemDataManager->handleWifiData(status, wifiName);
}

/*!
 * @brief Handle Temperature data.
 * @param temperature The temperature data.
 * @details This function processes the temperature data.
 */
void DataManager::handleTemperatureData(const QString &temperature)
{
	m_systemDataManager->handleTemperatureData(temperature);
}

/*!
 * @brief Handle IP Address data.
 * @param ipAddress The IP address.
 * @details This function processes the IP address data.
 */
void DataManager::handleIpAddressData(const QString &ipAddress)
{
	m_systemDataManager->handleIpAddressData(ipAddress);
}

/*!
 * @brief Handle Battery Percentage data.
 * @param batteryPercentage The battery percentage.
 * @details This function processes the battery percentage data.
 */
void DataManager::handleBatteryPercentage(float batteryPercentage)
{
	m_systemDataManager->handleBatteryPercentage(batteryPercentage);
}

/*!
 * @brief Toggle the driving mode.
 * @details This function toggles the driving mode between day and night mode.
 */
void DataManager::toggleDrivingMode()
{
	m_clusterSettingsManager->toggleDrivingMode();
}
/*!
 * @brief Toggle the cluster theme.
 * @details This function toggles the cluster theme between a light or dark
 * theme.
 */
void DataManager::toggleClusterTheme()
{
	m_clusterSettingsManager->toggleClusterTheme();

}
/*!
 * @brief Toggle the cluster metrics.
 * @details This function toggles the cluster metrics between kilometers and miles
 * by delegating the operation to the ClusterSettingsManager.
 */

void DataManager::toggleClusterMetrics()
{
	m_clusterSettingsManager->toggleClusterMetrics();
}

void DataManager::handleInferenceFrame(const std::vector<uchar> &jpegData) {
    QImage image;
    image.loadFromData(jpegData.data(), static_cast<int>(jpegData.size()), "JPG");
    emit inferenceImageReceived(image);
}
