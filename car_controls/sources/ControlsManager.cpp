#include "ControlsManager.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <QDebug>

ControlsManager::ControlsManager(int argc, char **argv, QObject *parent)
    : QObject(parent), m_engineController(0x40, 0x60, this),
      m_manualController(nullptr), m_currentMode(DrivingMode::Manual),
      m_manualControllerThread(nullptr), m_processMonitorThread(nullptr),
      m_carDataThread(nullptr), m_clientThread(nullptr), m_threadRunning(true) {

  // Initialize the joystick controller with callbacks
  m_manualController = new JoysticksController(
      [this](int steering) {
        if (m_currentMode == DrivingMode::Manual) {
          m_engineController.set_steering(steering);
        }
      },
      [this](int speed) {
        if (m_currentMode == DrivingMode::Manual) {
          m_engineController.set_speed(speed);
        }
      });

  if (!m_manualController->init()) {
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


  // Server Middleware Thread
  /* m_carDataThread = QThread::create([this, argc, argv]() {
    while (m_threadRunning) {
      m_carDataObject->runServer(argc, argv);
    }
  });
  m_carDataThread->start(); */


  // Client Middleware Interface Therad
  /* m_clientObject = new ClientThread();
  m_clientThread = QThread::create([this, argc, argv]() {
      m_clientObject->runClient(argc, argv);
  });
  m_clientThread->start(); */


  // **Process Monitoring Thread**
  m_processMonitorThread = QThread::create([this]() {
    QString targetProcessName = "HotWheels-app"; // Change this to actual process name

    while (m_threadRunning) {
      if (!isProcessRunning(targetProcessName)) {
        if (m_currentMode == DrivingMode::Automatic)
                setMode(DrivingMode::Manual);
        qDebug() << "Cluster is not running.";
      }
      QThread::sleep(1);  // Check every 1 second
    }
  });

  m_processMonitorThread->start();
}

ControlsManager::~ControlsManager() {
  // Stop the client thread safely
  if (m_clientThread) {
    m_clientThread->quit();
    m_clientThread->wait();
    delete m_clientThread;
  }

  // Stop the shared memory thread safely
  if (m_carDataThread) {
    m_threadRunning = false;
    m_carDataThread->quit();
    m_carDataThread->wait();
    delete m_carDataThread;
  }

  // Stop the process monitoring thread safely
  if (m_processMonitorThread) {
    m_threadRunning = false;
    m_processMonitorThread->quit();
    m_processMonitorThread->wait();
    delete m_processMonitorThread;
  }

  // Stop the controller thread safely
  if (m_manualControllerThread) {
    m_manualController->requestStop();
    m_manualControllerThread->quit();
    m_manualControllerThread->wait();
  }

  delete m_clientObject;
  delete m_manualController;
}

bool ControlsManager::isProcessRunning(const QString &processName) {
    QProcess process;
    process.start("pgrep", QStringList() << processName);
    process.waitForFinished();

    return !process.readAllStandardOutput().isEmpty();
}

void ControlsManager::readJoystickEnable()
{
  bool joystickData = m_clientObject->getJoystickValue();
  if (joystickData) {
    setMode(DrivingMode::Manual);
  } else {
    setMode(DrivingMode::Automatic);
  }
}

void ControlsManager::setMode(DrivingMode mode) {
  if (m_currentMode == mode)
    return;

  m_currentMode = mode;
}
