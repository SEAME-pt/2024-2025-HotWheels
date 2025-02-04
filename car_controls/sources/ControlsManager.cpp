#include "ControlsManager.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <QDebug>

ControlsManager::ControlsManager(QObject *parent)
    : QObject(parent), m_engineController(0x40, 0x60, this),
      m_manualController(nullptr), m_manualControllerThread(nullptr),
      m_currentMode(DrivingMode::Manual), m_sharedMemoryThread(nullptr),
      m_processMonitorThread(nullptr), m_threadRunning(true) {
  // Connect EngineController signals to ControlsManager signals
  connect(&m_engineController, &EngineController::directionUpdated, this,
          &ControlsManager::directionChanged);
  connect(&m_engineController, &EngineController::steeringUpdated, this,
          &ControlsManager::steeringChanged);

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

  // **Shared Memory Thread**
  m_sharedMemoryThread = QThread::create([this]() {
    while (m_threadRunning) {
      readSharedMemory();
      QThread::msleep(50);  // Adjust delay as needed
    }
  });

  m_sharedMemoryThread->start();

  // **Process Monitoring Thread**
  m_processMonitorThread = QThread::create([this]() {
    QString targetProcessName = "HotWheels-app"; // Change this to actual process name

    while (m_threadRunning) {
      if (!isProcessRunning(targetProcessName)) {
        if (m_currentMode == DrivingMode::Automatic)
                setMode(DrivingMode::Manual);
        qDebug() << "The monitored program has unexpectedly shut down!";
        //break;
      }
      QThread::sleep(1);  // Check every 1 second
    }
  });

  m_processMonitorThread->start();
}

ControlsManager::~ControlsManager() {

  // Stop the shared memory thread safely
  if (m_sharedMemoryThread) {
    m_threadRunning = false;
    m_sharedMemoryThread->quit();
    m_sharedMemoryThread->wait();
    delete m_sharedMemoryThread;
  }

  if (m_manualControllerThread) {
    m_manualController->requestStop();
    m_manualControllerThread->quit();
    m_manualControllerThread->wait();
  }

  delete m_manualController;
}

/* bool ControlsManager::isServiceRunning(const QString &serviceName) {
    QProcess process;
    process.start("systemctl", QStringList() << "is-active" << serviceName);
    process.waitForFinished();

    QString output = process.readAllStandardOutput().trimmed();
    return output == "active";  // Returns true if service is running
} */

bool ControlsManager::isProcessRunning(const QString &processName) {
    QProcess process;
    process.start("pgrep", QStringList() << processName);
    process.waitForFinished();

    return !process.readAllStandardOutput().isEmpty();
}

void ControlsManager::readSharedMemory() {
  int shm_fd = shm_open("/joystick_enable", O_RDWR, 0666);
  if (shm_fd == -1) {
      std::cerr << "Failed to open shared memory\n";
  }
  else {
    // Map shared memory
    void* ptr = mmap(0, sizeof(bool), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "Failed to map memory\n";
    }
    else {
      // Read the bool value
      bool* flag = static_cast<bool*>(ptr);

      setMode(*flag ? DrivingMode::Manual : DrivingMode::Automatic);

      // Cleanup
      munmap(ptr, sizeof(bool));
    }
    close(shm_fd);
  }
}

void ControlsManager::setMode(DrivingMode mode) {
  if (m_currentMode == mode)
    return;

  m_currentMode = mode;
}
