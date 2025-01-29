#include "ControlsManager.hpp"
#include <QDebug>

ControlsManager::ControlsManager(QObject *parent)
    : QObject(parent)
    , m_engineController(0x40, 0x60, this)
    , m_manualController(nullptr)
    , m_manualControllerThread(nullptr)
    , m_currentMode(DrivingMode::Manual)
{
    // Connect EngineController signals to ControlsManager signals
    connect(&m_engineController,
            &EngineController::directionUpdated,
            this,
            &ControlsManager::directionChanged);
    connect(&m_engineController,
            &EngineController::steeringUpdated,
            this,
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

    connect(m_manualControllerThread,
            &QThread::started,
            m_manualController,
            &JoysticksController::processInput);
    connect(m_manualController,
            &JoysticksController::finished,
            m_manualControllerThread,
            &QThread::quit);

    m_manualControllerThread->start();
}

ControlsManager::~ControlsManager()
{
    if (m_manualControllerThread) {
        m_manualController->requestStop();
        m_manualControllerThread->quit();
        m_manualControllerThread->wait();
    }

    delete m_manualController;
}

void ControlsManager::setMode(DrivingMode mode)
{
    if (m_currentMode == mode)
        return;

    m_currentMode = mode;

    // if (m_currentMode == DrivingMode::Automatic) {
    //     qDebug() << "Switched to Automatic mode (joystick disabled).";
    // } else if (m_currentMode == DrivingMode::Manual) {
    //     qDebug() << "Switched to Manual mode (joystick enabled).";
    // }
}

void ControlsManager::drivingModeUpdated(DrivingMode newMode)
{
    setMode(newMode);
}
