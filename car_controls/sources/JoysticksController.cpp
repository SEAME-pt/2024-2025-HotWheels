#include "JoysticksController.hpp"
#include <QDebug>
#include <QThread>

JoysticksController::JoysticksController(std::function<void(int)> steeringCallback,
                                         std::function<void(int)> speedCallback,
                                         QObject *parent)
    : QObject(parent)
    , m_joystick(nullptr)
    , m_updateSteering(std::move(steeringCallback))
    , m_updateSpeed(std::move(speedCallback))
    , m_running(false)
{}

JoysticksController::~JoysticksController()
{
    if (m_joystick) {
        SDL_JoystickClose(m_joystick);
    }
    SDL_Quit();
}

bool JoysticksController::init()
{
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        qDebug() << "Failed to initialize SDL:" << SDL_GetError();
        return false;
    }

    m_joystick = SDL_JoystickOpen(0);
    if (!m_joystick) {
        init();
        /* qDebug() << "Failed to open joystick.";
        SDL_Quit();
        return false; */
    }

    return true;
}

void JoysticksController::requestStop()
{
    m_running = false;
}

void JoysticksController::processInput()
{
    m_running = true;

    if (!m_joystick) {
        qDebug() << "Joystick not initialized.";
        emit finished();
        return;
    }

    while (m_running && !QThread::currentThread()->isInterruptionRequested()) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_JOYAXISMOTION) {
                if (e.jaxis.axis == 0) {
                    m_updateSteering(static_cast<int>(e.jaxis.value / 32767.0 * 180));
                } else if (e.jaxis.axis == 3) {
                    m_updateSpeed(static_cast<int>(e.jaxis.value / 32767.0 * 100));
                }
            }
        }
        QThread::msleep(10);
    }

    // qDebug() << "Joystick controller loop finished.";
    emit finished();
}
