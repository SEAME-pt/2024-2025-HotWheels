/*!
 * @file JoysticksController.hpp
 * @brief Definition of the JoysticksController class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the JoysticksController class,
 * which is responsible for controlling the car's steering and speed using a
 * joystick.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef JOYSTICKS_CONTROLLER_HPP
#define JOYSTICKS_CONTROLLER_HPP

#include <QObject>
#include <SDL2/SDL.h>
#include <functional>

/*!
 * @brief Class that controls the car's steering and speed using a joystick.
 * @class JoysticksController inherits from QObject
 */
class JoysticksController : public QObject {
  Q_OBJECT

private:
  /*! @brief Pointer to the joystick device. */
  SDL_Joystick *m_joystick;
  /*! @brief Callback function to update the steering value. */
  std::function<void(int)> m_updateSteering;
  /*! @brief Callback function to update the speed value. */
  std::function<void(int)> m_updateSpeed;
  /*! @brief Flag to indicate if the controller is running. */
  bool m_running;

public:
  JoysticksController(std::function<void(int)> steeringCallback,
                      std::function<void(int)> speedCallback,
                      QObject *parent = nullptr);
  ~JoysticksController();
  bool init();
  void requestStop();

  slots:
  void processInput();

signals:
  /*! @brief Signal emitted when the controller is finished. */
  void finished();
};

#endif // JOYSTICKS_CONTROLLER_HPP
