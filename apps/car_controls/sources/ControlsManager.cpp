/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the ControlsManager class,
 * which is responsible for managing the different controllers and worker threads
 * for the car controls.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <QDebug>

/*!
 * @brief Constructs a ControlsManager object.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param parent The parent QObject for this ControlsManager.
 * @details Initializes the engine controller, joystick controller, and various
 * worker threads for managing car controls. Sets up joystick control with
 * callbacks for steering and speed adjustments, manages server and client
 * middleware threads, monitors processes, and handles joystick enable status
 * through dedicated threads.
 */
ControlsManager::ControlsManager(int argc, char **argv, QObject *parent)
	: QObject(parent), m_engineController(0x40, 0x60, this),
	  m_manualController(nullptr), m_currentMode(DrivingMode::Manual),
	  m_subscriberObject(nullptr), m_manualControllerThread(nullptr),
	  m_subscriberThread(nullptr), m_joystickControlThread(nullptr),
	  m_cameraStreamerThread(nullptr), m_cameraStreamerObject(nullptr)
{

	// Initialize the joystick controller with callbacks
	m_manualController = new JoysticksController(
		[this](int steering)
		{
			if (m_currentMode == DrivingMode::Manual)
			{
				m_engineController.set_steering(steering);
			}
		},
		[this](int speed)
		{
			if (m_currentMode == DrivingMode::Manual)
			{
				m_engineController.set_speed(speed);
			}
		});

	if (!m_manualController->init())
	{
		qDebug() << "Failed to initialize joystick controller.";
		return;
	}

	// Start the joystick controller in its own thread
	m_manualControllerThread = new QThread(this);
	m_manualController->moveToThread(m_manualControllerThread);

	connect(m_manualControllerThread, &QThread::started, m_manualController,
			&JoysticksController::processInput);
	connect(m_manualController, &JoysticksController::finished,
			m_manualControllerThread, &QThread::quit);

	m_manualControllerThread->start();

	// **Client Middleware Interface Thread**
	m_subscriberObject = new Subscriber();
	m_subscriberThread = QThread::create([this, argc, argv]()
									{
		m_subscriberObject->connect("tcp://localhost:5555");
		m_subscriberObject->subscribe("joystick_value");
		while (m_running) {
			zmq::message_t message;
			m_subscriberObject->getSocket().recv(&message, 0);

			std::string received_msg(static_cast<char*>(message.data()), message.size());

			if (received_msg.find("joystick_value") == 0) {
				std::string value = received_msg.substr(std::string("joystick_value ").length());
				if (value == "true") {
					setMode(DrivingMode::Manual);
				}
				else if (value == "false") {
					setMode(DrivingMode::Automatic);
				}
			}
		}
	});
	m_subscriberThread->start();

	// **Running inference Thread**
	m_cameraStreamerThread = QThread::create([this, argc, argv]()
									{
		try {
			std::cout << "Starting TensorRT Inference on Jetson..." << std::endl;

			// Path to your TensorRT engine file - adjust path as needed for Jetson
			std::string enginePath = "/home/hotweels/dev/model_loader/models/model.engine";

			// Create the TensorRT inferencer
			std::cout << "Loading TensorRT engine from: " << enginePath << std::endl;
			//TensorRTInferencer inferencer(enginePath);

			auto inferencer = std::make_shared<TensorRTInferencer>(enginePath);

			// Create the camera streamer with the inferencer
			std::cout << "Initializing CSI camera..." << std::endl;
			m_cameraStreamerObject = new CameraStreamer(inferencer, 0.5, "Jetson Camera Inference", true);
			m_cameraStreamerObject->start();
		} catch (const std::exception& e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}

		std::cout << "Shutting down..." << std::endl;
	});
	connect(m_cameraStreamerThread, &QThread::finished, m_cameraStreamerThread, &QObject::deleteLater);
	m_cameraStreamerThread->start();
}


/*!
 * @brief Destructor for the ControlsManager class.
 * @details Safely stops and cleans up all threads and resources associated
 *          with the ControlsManager. This includes stopping the client,
 *          shared memory, process monitoring, joystick control, and manual
 *          controller threads. It also deletes associated objects such as
 *          m_carDataObject, m_subscriberThread, and m_manualController.
 */

ControlsManager::~ControlsManager()
{
	// Stop the client thread safely
	if (m_subscriberThread) {
		if (m_subscriberObject) {
			m_subscriberObject->stop();
		}

		m_subscriberThread->wait();
		delete m_subscriberThread;
		m_subscriberThread = nullptr;
	}


	// Stop manual controller thread
	if (m_manualControllerThread) {
		if (m_manualController)
			m_manualController->requestStop();

		m_manualControllerThread->quit();
		m_manualControllerThread->wait();
		delete m_manualControllerThread;
		m_manualControllerThread = nullptr;
	}

	// Stop camera streamer thread
	qDebug() << "[Shutdown] Stopping camera streamer thread...";
	if (m_cameraStreamerThread) {
		if (m_cameraStreamerObject)
			m_cameraStreamerObject->stop();

		m_cameraStreamerThread->quit();
		m_cameraStreamerThread->wait();
		delete m_cameraStreamerThread;
		m_cameraStreamerThread = nullptr;
	}
	// Clean up objects
	qDebug() << "[Shutdown] Deleting m_cameraStreamerObject...";
	delete m_cameraStreamerObject;
	m_cameraStreamerObject = nullptr;

	qDebug() << "[Shutdown] Deleting m_manualController...";
	delete m_manualController;
	m_manualController = nullptr;

	qDebug() << "[Shutdown] Deleting m_subscriberObject...";
	delete m_subscriberObject;
	m_subscriberObject = nullptr;

	qDebug() << "[Shutdown] Cleanup complete.";
}

/*!
 * @brief Sets the driving mode.
 * @param mode The new driving mode.
 * @details Updates the current driving mode if it has changed.
 */
void ControlsManager::setMode(DrivingMode mode)
{
	if (m_currentMode == mode)
		return;

	m_currentMode = mode;
}
