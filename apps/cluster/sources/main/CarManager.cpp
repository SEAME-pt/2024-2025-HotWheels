/*!
 * @file CarManager.cpp
 * @brief Implementation of the CarManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the implementation of the CarManager class, which
 * is used to manage the entire system.
 * @note This class is used to manage the entire system, including the data, CAN
 * bus, controls, display, system, and mileage.
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
#include <QString>

/*!
 * @brief Construct a new CarManager object.
 * @param parent The parent QWidget.
 * @details This constructor initializes the CarManager object with the
 * specified parent.
 */
CarManager::CarManager(int argc, char **argv, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CarManager)
    , m_running(true)
    , m_dataManager(new DataManager())
    , m_canBusManager(new CanBusManager("/dev/spidev0.0"))
    , m_controlsManager(new ControlsManager(argc, argv))
    , m_displayManager(nullptr)
    , m_systemManager(new SystemManager())
    , m_mileageManager(new MileageManager("/home/hotweels/app_data/mileage.json"))
    , m_inferenceSubscriber(nullptr)
    , m_inferenceSubscriberThread(nullptr)
{
    ui->setupUi(this);

    QString style = R"(
      QMainWindow {
          background-image: url(:/images/background.jpg);
          background-repeat: no-repeat;
          background-position: center;
          background-size: 110% 110%;
      }
    )";
    this->setStyleSheet(style);

    initializeComponents();

    m_inferenceSubscriber = new Subscriber();
    m_inferenceSubscriberThread = QThread::create([this]() {
      m_inferenceSubscriber->connect("tcp://localhost:5556");  // Your image port
      m_inferenceSubscriber->subscribe("inference_frame");

      while (m_running) {
        try {
          zmq::pollitem_t items[] = {
            { static_cast<void*>(m_inferenceSubscriber->getSocket()), 0, ZMQ_POLLIN, 0 }
          };

          zmq::poll(items, 1, 100);  // Timeout: 100 ms

          if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if (!m_inferenceSubscriber->getSocket().recv(&message, 0)) {
              continue;
            }

            std::string received_msg(static_cast<char*>(message.data()), message.size());
            //std::cout << "[Subscriber] Raw message: " << received_msg.substr(0, 30) << "... (" << message.size() << " bytes)" << std::endl;

            const std::string topic = "inference_frame ";
            if (received_msg.find(topic) == 0) {
              std::vector<uchar> jpegData(
                received_msg.begin() + topic.size(),
                received_msg.end()
              );
              m_dataManager->handleInferenceFrame(jpegData);
            }
          }
        } catch (const zmq::error_t& e) {
          std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
          break;
        }
      }
    });
    m_inferenceSubscriberThread->start();
}

CarManager::~CarManager()
{
    m_running = false;
    if (m_inferenceSubscriberThread) {
        m_inferenceSubscriber->stop();
        m_inferenceSubscriberThread->quit();
        m_inferenceSubscriberThread->wait();
        delete m_inferenceSubscriberThread;
        m_inferenceSubscriberThread = nullptr;
    }
    delete m_inferenceSubscriber;
    m_inferenceSubscriber = nullptr;

    delete m_displayManager;
    delete m_controlsManager;
    delete m_canBusManager;
    delete m_dataManager;
    delete m_mileageManager;
    delete m_systemManager;
    delete ui;
}

/*!
 * @brief Initialize the components of the CarManager.
 * @details This function initializes the DataManager, CanBusManager,
 * ControlsManager, DisplayManager, SystemManager, and MileageManager.
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

/*!
 * @brief Initialize the DataManager.
 * @details !No additional logic for now; ready for future extensions.
 */
void CarManager::initializeDataManager() {
  // No additional logic for now; ready for future extensions
}

/*!
 * @brief Initialize the CanBusManager.
 * @details This function initializes the CanBusManager and connects its signals
 * to the DataManager slots.
 */
void CarManager::initializeCanBusManager() {
  if (m_canBusManager->initialize()) {
    connect(m_canBusManager, &CanBusManager::speedUpdated, m_dataManager,
	    &DataManager::handleSpeedData);

    connect(m_canBusManager, &CanBusManager::rpmUpdated, m_dataManager,
	    &DataManager::handleRpmData);
  }
}

/*!
 * @brief Initialize the ControlsManager.
 * @details This function initializes the ControlsManager and connects its
 * signals to the DataManager slots.
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

/*!
 * @brief Initialize the DisplayManager.
 * @details This function initializes the DisplayManager and connects its
 * signals to the DataManager slots.
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

    connect(m_dataManager, &DataManager::systemTemperatureUpdated,
	    m_displayManager, &DisplayManager::updateTemperature);

    connect(m_dataManager, &DataManager::batteryPercentageUpdated,
	    m_displayManager, &DisplayManager::updateBatteryPercentage);

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

    connect(m_dataManager, &DataManager::inferenceImageReceived,
      m_displayManager, &DisplayManager::displayInferenceImage);
  }
}

/*!
 * @brief Initialize the SystemManager.
 * @details This function initializes the SystemManager and connects its signals
 * to the DataManager slots.
 */
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
    }
}

/*!
 * @brief Initialize the MileageManager.
 * @details This function initializes the MileageManager and connects its
 * signals to the DataManager slots.
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
