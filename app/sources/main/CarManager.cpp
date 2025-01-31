#include "CarManager.hpp"
#include <QDebug>
#include "ui_CarManager.h"

CarManager::CarManager(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CarManager)
    , m_dataManager(new DataManager())
    , m_canBusManager(new CanBusManager("/dev/spidev0.0"))
    , m_controlsManager(new ControlsManager())
    , m_displayManager(nullptr)
    , m_systemManager(new SystemManager())
    , m_mileageManager(new MileageManager("/home/hotweels/app_data/mileage.json"))
{
    ui->setupUi(this);
    initializeComponents();
}

CarManager::~CarManager()
{
    delete m_displayManager;
    delete m_controlsManager;
    delete m_canBusManager;
    delete m_dataManager;
    delete m_mileageManager;
    delete m_systemManager;
    delete ui;
}

void CarManager::initializeComponents()
{
    initializeDataManager();
    initializeCanBusManager();
    initializeControlsManager();
    initializeDisplayManager();
    initializeSystemManager();
    initializeMileageManager();
    qDebug() << "[Main] HotWheels Cluster operational.";
}

void CarManager::initializeDataManager()
{
    // No additional logic for now; ready for future extensions
}

void CarManager::initializeCanBusManager()
{
    if (m_canBusManager->initialize()) {
        connect(m_canBusManager,
                &CanBusManager::speedUpdated,
                m_dataManager,
                &DataManager::handleSpeedData);

        connect(m_canBusManager,
                &CanBusManager::rpmUpdated,
                m_dataManager,
                &DataManager::handleRpmData);
    }
}

void CarManager::initializeControlsManager()
{
    if (m_controlsManager) {
        // Connect ControlsManager signals to DataManager slots
        connect(m_controlsManager,
                &ControlsManager::directionChanged,
                m_dataManager,
                &DataManager::handleDirectionData);

        connect(m_controlsManager,
                &ControlsManager::steeringChanged,
                m_dataManager,
                &DataManager::handleSteeringData);

        connect(m_dataManager,
                &DataManager::drivingModeUpdated,
                m_controlsManager,
                &ControlsManager::drivingModeUpdated);
    }
}

void CarManager::initializeDisplayManager()
{
    if (ui) {
        m_displayManager = new DisplayManager(ui, this);

        // Connect DataManager signals to DisplayManager slots
        connect(m_dataManager,
                &DataManager::canDataProcessed,
                m_displayManager,
                &DisplayManager::updateCanBusData);

        connect(m_dataManager,
                &DataManager::engineDataProcessed,
                m_displayManager,
                &DisplayManager::updateEngineData);

        connect(m_dataManager,
                &DataManager::systemTimeUpdated,
                m_displayManager,
                &DisplayManager::updateSystemTime);

        connect(m_dataManager,
                &DataManager::systemWifiUpdated,
                m_displayManager,
                &DisplayManager::updateWifiStatus);

        connect(m_dataManager,
                &DataManager::systemTemperatureUpdated,
                m_displayManager,
                &DisplayManager::updateTemperature);

        connect(m_dataManager,
                &DataManager::batteryPercentageUpdated,
                m_displayManager,
                &DisplayManager::updateBatteryPercentage);

        connect(m_dataManager,
                &DataManager::ipAddressUpdated,
                m_displayManager,
                &DisplayManager::updateIpAddress);

        connect(m_dataManager,
                &DataManager::drivingModeUpdated,
                m_displayManager,
                &DisplayManager::updateDrivingMode);

        connect(m_dataManager,
                &DataManager::clusterThemeUpdated,
                m_displayManager,
                &DisplayManager::updateClusterTheme);

        connect(m_dataManager,
                &DataManager::clusterMetricsUpdated,
                m_displayManager,
                &DisplayManager::updateClusterMetrics);

        connect(m_dataManager,
                &DataManager::mileageUpdated,
                m_displayManager,
                &DisplayManager::updateMileage);

        // Connect DisplayManager toggle signals to DataManager slots
        connect(m_displayManager,
                &DisplayManager::drivingModeToggled,
                m_dataManager,
                &DataManager::toggleDrivingMode);

        connect(m_displayManager,
                &DisplayManager::clusterThemeToggled,
                m_dataManager,
                &DataManager::toggleClusterTheme);

        connect(m_displayManager,
                &DisplayManager::clusterMetricsToggled,
                m_dataManager,
                &DataManager::toggleClusterMetrics);
    }
}

void CarManager::initializeSystemManager()
{
    if (m_systemManager) {
        m_systemManager->initialize();
        // Connect SystemManager signals to DataManager slots
        connect(m_systemManager,
                &SystemManager::timeUpdated,
                m_dataManager,
                &DataManager::handleTimeData);

        connect(m_systemManager,
                &SystemManager::wifiStatusUpdated,
                m_dataManager,
                &DataManager::handleWifiData);

        connect(m_systemManager,
                &SystemManager::temperatureUpdated,
                m_dataManager,
                &DataManager::handleTemperatureData);

        // Connect SystemManager's battery signal to DataManager's battery slot
        connect(m_systemManager,
                &SystemManager::batteryPercentageUpdated,
                m_dataManager,
                &DataManager::handleBatteryPercentage);

        connect(m_systemManager,
                &SystemManager::ipAddressUpdated,
                m_dataManager,
                &DataManager::handleIpAddressData);
    }
}

void CarManager::initializeMileageManager()
{
    if (m_mileageManager) {
        m_mileageManager->initialize();

        // Connect CanBusManager signals to MileageManager slots
        connect(m_canBusManager,
                &CanBusManager::speedUpdated,
                m_mileageManager,
                &MileageManager::onSpeedUpdated);

        // Connect MileageManager signals to DataManager slots
        connect(m_mileageManager,
                &MileageManager::mileageUpdated,
                m_dataManager,
                &DataManager::handleMileageUpdate);
    }
}
