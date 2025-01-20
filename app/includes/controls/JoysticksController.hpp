#ifndef JOYSTICKS_CONTROLLER_HPP
#define JOYSTICKS_CONTROLLER_HPP

#include <QObject>
#include <SDL2/SDL.h>
#include <functional>

class JoysticksController : public QObject
{
    Q_OBJECT

private:
    SDL_Joystick *m_joystick;
    std::function<void(int)> m_updateSteering;
    std::function<void(int)> m_updateSpeed;
    bool m_running;

public:
    explicit JoysticksController(std::function<void(int)> steeringCallback,
                                 std::function<void(int)> speedCallback,
                                 QObject *parent = nullptr);
    ~JoysticksController();

    bool init();
    void requestStop();

public slots:
    void processInput();

signals:
    void finished();
};

#endif // JOYSTICKS_CONTROLLER_HPP
