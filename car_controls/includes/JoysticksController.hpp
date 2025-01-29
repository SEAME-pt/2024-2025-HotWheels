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
    /**
     * Constructor for the JoysticksController class, initializing the joystick controller with the provided callbacks for steering and speed.
     *
     * @param steeringCallback The callback function to update the steering value.
     * @param speedCallback The callback function to update the speed value.
     * @param parent The parent QObject for this instance.
     */
    JoysticksController(std::function<void(int)> steeringCallback,
                        std::function<void(int)> speedCallback,
                        QObject *parent = nullptr);

    /**
     * Destructor for the JoysticksController class, cleaning up the joystick resources.
     */
    ~JoysticksController();

    /**
     * Initializes the joystick controller by setting up SDL and opening the joystick device.
     *
     * @return True if the joystick was successfully initialized, false otherwise.
     */
    bool init();

    /**
     * Requests the joystick controller to stop processing input.
     */
    void requestStop();

public slots:
    /**
     * Processes joystick input in a loop, updating steering and speed based on joystick events.
     * This method runs in a separate thread.
     */
    void processInput();

signals:
    /**
     * Emitted when the joystick input processing has finished.
     */
    void finished();
};

#endif // JOYSTICKS_CONTROLLER_HPP
