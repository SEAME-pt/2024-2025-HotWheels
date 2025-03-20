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
	  m_processMonitorThread(nullptr), m_subscriberThread(nullptr),
	  m_joystickControlThread(nullptr), m_threadRunning(true)
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
		while (true) {
			zmq::message_t message;
			m_subscriberObject->getSocket().recv(&message, 0);

			std::string received_msg(static_cast<char*>(message.data()), message.size());
			std::cout << "Received: " << received_msg << std::endl;

			if (received_msg.find("joystick_value") == 0) {
				std::string value = received_msg.substr(std::string("joystick_value ").length());
				if (value == "true")
				{
					qDebug() << "Received true" << std::endl;
					setMode(DrivingMode::Manual);
				}
				else if (value == "false")
				{
					qDebug() << "Received false" << std::endl;
					setMode(DrivingMode::Automatic);
				}
			}
		}
		//m_subscriberObject->listen();
	});
	m_subscriberThread->start();

	// **Process Monitoring Thread**
	m_processMonitorThread = QThread::create([this]()
											 {
	QString targetProcessName = "HotWheels-app"; // Change this to actual process name

	while (m_threadRunning) {
	  if (!isProcessRunning(targetProcessName)) {
		if (m_currentMode == DrivingMode::Automatic)
				setMode(DrivingMode::Manual);
		//qDebug() << "Cluster is not running.";
	  }
	  QThread::sleep(1);  // Check every 1 second
	} });
	m_processMonitorThread->start();

	// **Joystick Control Thread**
	/* m_joystickControlThread = QThread::create([this]()
											  {
	while (m_threadRunning) {
	  readJoystickEnable();
	  QThread::msleep(100);  // Adjust delay as needed
	} });
	m_joystickControlThread->start(); */
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
	if (m_subscriberThread)
	{
		m_subscriberObject->stop();
		m_subscriberThread->quit();
		m_subscriberThread->wait();
		delete m_subscriberThread;
	}

	// Stop the process monitoring thread safely
	if (m_processMonitorThread)
	{
		m_threadRunning = false;
		m_processMonitorThread->quit();
		m_processMonitorThread->wait();
		delete m_processMonitorThread;
	}

	// Stop the controller thread safely
	if (m_manualControllerThread)
	{
		m_manualController->requestStop();
		m_manualControllerThread->quit();
		m_manualControllerThread->wait();
		delete m_manualControllerThread;
	}

	// Stop the joystick control thread safely
	if (m_joystickControlThread)
	{
		m_joystickControlThread->quit();
		m_joystickControlThread->wait();
		delete m_joystickControlThread;
	}

	delete m_subscriberThread;
	delete m_manualController;
}

/*!
 * @brief Check if a process is running.
 * @param processName The name of the process to check.
 * @return True if the process is running, false otherwise.
 * @details Uses the `pgrep` command to determine if a given process is active.
 */
bool ControlsManager::isProcessRunning(const QString &processName)
{
	QProcess process;
	process.start("pgrep", QStringList() << processName);
	process.waitForFinished();

	return !process.readAllStandardOutput().isEmpty();
}

/*!
 * @brief Reads joystick enable status.
 * @details Checks if joystick control is enabled through the client middleware
 *          and updates the driving mode accordingly.
 */
/* void ControlsManager::readJoystickEnable()
{
	bool joystickData = m_subscriberThread->getJoystickValue();
	if (joystickData)
	{
		setMode(DrivingMode::Manual);
	}
	else
	{
		setMode(DrivingMode::Automatic);
	}
} */

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
