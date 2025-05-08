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
    initializeComponents();

    m_inferenceSubscriber = new Subscriber();
  m_inferenceSubscriberThread = QThread::create([this]() {
      // 1. Connect before subscribing
      m_inferenceSubscriber->connect("tcp://localhost:5556");
      qDebug() << "[Subscriber] Connected to publisher";

      // 2. Subscribe to exact topic - make sure there's no whitespace or hidden characters
      const std::string topic = "inference_frame";
      m_inferenceSubscriber->subscribe(topic);
      qDebug() << "[Subscriber] Subscribed to topic with length:" << topic.length()
              << "Topic bytes:" << QString::fromStdString(topic);

      // Get and set socket options for debugging
      int hwm;
      size_t hwm_size = sizeof(hwm);
      zmq_getsockopt(m_inferenceSubscriber->getSocketHandle(), ZMQ_RCVHWM, &hwm, &hwm_size);
      qDebug() << "[Subscriber] Current HWM setting:" << hwm;

      // Increase HWM if needed
      hwm = 1000;  // Higher value to store more messages
      zmq_setsockopt(m_inferenceSubscriber->getSocketHandle(), ZMQ_RCVHWM, &hwm, sizeof(hwm));
      qDebug() << "[Subscriber] Updated HWM setting to:" << hwm;

      // 3. Give some time for the subscription to register
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));

      while (m_running) {
            zmq::message_t topic_msg;

            // Poll with timeout to avoid blocking indefinitely
            // Ensure we're using the socket correctly
            void* socket_ptr = static_cast<void*>(m_inferenceSubscriber->getSocketHandle());
            zmq::pollitem_t items[] = {
                { socket_ptr, 0, ZMQ_POLLIN, 0 }
            };

            // Debug message to show we're polling
            static int poll_count = 0;
            if (++poll_count % 10 == 0) {  // Only print every 10th time to avoid flooding
                qDebug() << "[Subscriber] Polling for messages...";

            // Poll with timeout of 100ms
            zmq::poll(items, 1, 100);

            if (items[0].revents & ZMQ_POLLIN) {
                qDebug() << "[Subscriber] Detected incoming message!";

                // Receive the topic (first part of the message)
                try {
                    zmq::message_t topic_msg;
                    bool received = false;

                    // Try with timeout to avoid hanging
                    try {
                        received = m_inferenceSubscriber->getSocket().recv(&topic_msg, ZMQ_DONTWAIT);
                        if (!received) {
                            qDebug() << "[Subscriber] No message available (EAGAIN)";
                            continue;
                        }
                    } catch (const zmq::error_t& e) {
                        qDebug() << "[Subscriber] Error receiving topic:" << e.what();
                        continue;
                    }

                    // Successfully received topic
                    std::string topic_str(static_cast<char*>(topic_msg.data()), topic_msg.size());
                    qDebug() << "[Subscriber] Topic received:" << QString::fromStdString(topic_str)
                            << "(length:" << topic_str.length() << ")";

                    // Topic received, now get the data part
                    zmq::message_t image_msg;
                    try {
                        received = m_inferenceSubscriber->getSocket().recv(&image_msg, ZMQ_DONTWAIT);
                        if (!received) {
                            qDebug() << "[Subscriber] No image part available";
                            continue;
                        }
                    } catch (const zmq::error_t& e) {
                        qDebug() << "[Subscriber] Error receiving image:" << e.what();
                        continue;
                    }

                    qDebug() << "[Subscriber] Received topic:" << QString::fromStdString(topic_str)
                            << ", size:" << image_msg.size();

                    if (topic_str == "inference_frame" && image_msg.size() > 0) {
                        std::vector<uchar> jpegData(
                            static_cast<uchar*>(image_msg.data()),
                            static_cast<uchar*>(image_msg.data()) + image_msg.size()
                        );

                        // Process the received image data
                        QMetaObject::invokeMethod(this, [this, jpegData]() {
                            m_dataManager->handleInferenceFrame(jpegData);
                        }, Qt::QueuedConnection);
                    }
                } catch (const zmq::error_t& e) {
                    qDebug() << "[Subscriber] ZMQ error: " << e.what();
                }
            }
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

	connect(m_systemManager,
		&SystemManager::ipAddressUpdated,
		m_dataManager,
		&DataManager::handleIpAddressData);
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
