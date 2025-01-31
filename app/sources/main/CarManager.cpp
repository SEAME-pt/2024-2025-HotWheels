/**
 * @file CarManager.cpp
 * @brief Implementation of the CarManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the CarManager class, which is used to manage the entire system.
 * @note This class is used to manage the entire system, including the data, CAN bus, controls, display, system, and mileage.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that all components are properly initialized and connected.
 * @see CarManager.hpp
 * @copyright Copyright (c) 2025
 */

#include "CarManager.hpp"
#include "ui_CarManager.h"
#include <QDebug>

/**
 * @brief Construct a new CarManager object.
 * @param parent The parent QWidget.
 * @details This constructor initializes the CarManager object with the specified parent.
 */
CarManager::CarManager(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::CarManager),
      m_dataManager(new DataManager()),
      m_canBusManager(new CanBusManager("/dev/spidev0.0")),
      m_controlsManager(new ControlsManager()), m_displayManager(nullptr),
      m_systemManager(new SystemManager(this)),
      m_mileageManager(
          new MileageManager("/home/hotweels/app_data/mileage.json")) {
  ui->setupUi(this);
  initializeComponents();
}

/**
 * @brief Destroy the CarManager object.
 * @details This destructor cleans up the resources used by the CarManager.
 */
CarManager::~CarManager() {
  delete m_displayManager;
  delete m_controlsManager;
  delete m_canBusManager;
  delete m_dataManager;
  delete m_mileageManager;
  delete ui;
}

/**
 * @brief Initialize the components of the CarManager.
 * @details This function initializes the DataManager, CanBusManager, ControlsManager, DisplayManager, SystemManager, and MileageManager.
 */
void CarManager::initializeComponents() {
  initializeDataManager();
  initializeCanBusManager();
  initializeControlsManager();
  initializeDisplayManager();
  initializeSystemManager();
  initializeMileageManager();
  qDebug() << "[Main] HotWheels Cluster operational.";
}

/**
 * @brief Initialize the DataManager.
 * @details !No additional logic for now; ready for future extensions.
 */
void CarManager::initializeDataManager() {
  // No additional logic for now; ready for future extensions
}

/**
 * @brief Initialize the CanBusManager.
 * @details This function initializes the CanBusManager and connects its signals to the DataManager slots.
 */
void CarManager::initializeCanBusManager() {
  if (m_canBusManager->initialize()) {
    connect(m_canBusManager, &CanBusManager::speedUpdated, m_dataManager,
            &DataManager::handleSpeedData);

    connect(m_canBusManager, &CanBusManager::rpmUpdated, m_dataManager,
            &DataManager::handleRpmData);
  }
}

/**
 * @brief Initialize the ControlsManager.
 * @details This function initializes the ControlsManager and connects its signals to the DataManager slots.
 */
void CarManager::initializeControlsManager() {
  if (m_controlsManager) {
    // Connect ControlsManager signals to DataManager slots
    connect(m_controlsManager, &ControlsManager::directionChanged,
            m_dataManager, &DataManager::handleDirectionData);

    connect(m_controlsManager, &ControlsManager::steeringChanged, m_dataManager,
            &DataManager::handleSteeringData);

    connect(m_dataManager, &DataManager::drivingModeUpdated, m_controlsManager,
            &ControlsManager::drivingModeUpdated);
  }
}

/**
 * @brief Initialize the DisplayManager.
 * @details This function initializes the DisplayManager and connects its signals to the DataManager slots.
 */
void CarManager::initializeDisplayManager() {
  if (ui) {
    m_displayManager = new DisplayManager(ui, this);

    // Connect DataManager signals to DisplayManager slots
    connect(m_dataManager, &DataManager::canDataProcessed, m_displayManager,
            &DisplayManager::updateCanBusData);

    connect(m_dataManager, &DataManager::engineDataProcessed, m_displayManager,
            &DisplayManager::updateEngineData);

    connect(m_dataManager, &DataManager::systemTimeUpdated, m_displayManager,
            &DisplayManager::updateSystemTime);

    connect(m_dataManager, &DataManager::systemWifiUpdated, m_displayManager,
            &DisplayManager::updateWifiStatus);

    connect(m_dataManager, &DataManager::systemTemperatureUpdated,
            m_displayManager, &DisplayManager::updateTemperature);

    connect(m_dataManager, &DataManager::batteryPercentageUpdated,
            m_displayManager, &DisplayManager::updateBatteryPercentage);

    connect(m_dataManager, &DataManager::ipAddressUpdated, m_displayManager,
            &DisplayManager::updateIpAddress);

    connect(m_dataManager, &DataManager::drivingModeUpdated, m_displayManager,
            &DisplayManager::updateDrivingMode);

    connect(m_dataManager, &DataManager::clusterThemeUpdated, m_displayManager,
            &DisplayManager::updateClusterTheme);

    connect(m_dataManager, &DataManager::clusterMetricsUpdated,
            m_displayManager, &DisplayManager::updateClusterMetrics);

    connect(m_dataManager, &DataManager::mileageUpdated, m_displayManager,
            &DisplayManager::updateMileage);

    // Connect DisplayManager toggle signals to DataManager slots
    connect(m_displayManager, &DisplayManager::drivingModeToggled,
            m_dataManager, &DataManager::toggleDrivingMode);

    connect(m_displayManager, &DisplayManager::clusterThemeToggled,
            m_dataManager, &DataManager::toggleClusterTheme);

    connect(m_displayManager, &DisplayManager::clusterMetricsToggled,
            m_dataManager, &DataManager::toggleClusterMetrics);
  }
}

/**
 * @brief Initialize the SystemManager.
 * @details This function initializes the SystemManager and connects its signals to the DataManager slots.
 */
void CarManager::initializeSystemManager() {
  if (m_systemManager) {
    // Connect SystemManager signals to DataManager slots
    connect(m_systemManager, &SystemManager::timeUpdated, m_dataManager,
            &DataManager::handleTimeData);

    connect(m_systemManager, &SystemManager::wifiStatusUpdated, m_dataManager,
            &DataManager::handleWifiData);

    connect(m_systemManager, &SystemManager::temperatureUpdated, m_dataManager,
            &DataManager::handleTemperatureData);

    // Connect SystemManager's battery signal to DataManager's battery slot
    connect(m_systemManager, &SystemManager::batteryPercentageUpdated,
            m_dataManager, &DataManager::handleBatteryPercentage);

    connect(m_systemManager, &SystemManager::ipAddressUpdated, m_dataManager,
            &DataManager::handleIpAddressData);
  }
}

/**
 * @brief Initialize the MileageManager.
 * @details This function initializes the MileageManager and connects its signals to the DataManager slots.
 */
void CarManager::initializeMileageManager() {
  if (m_mileageManager) {
    m_mileageManager->initialize();

    // Connect CanBusManager signals to MileageManager slots
    connect(m_canBusManager, &CanBusManager::speedUpdated, m_mileageManager,
            &MileageManager::onSpeedUpdated);

    // Connect MileageManager signals to DataManager slots
    connect(m_mileageManager, &MileageManager::mileageUpdated, m_dataManager,
            &DataManager::handleMileageUpdate);
  }
}
