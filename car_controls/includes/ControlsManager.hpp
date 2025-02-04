#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include <QObject>
#include <QThread>
#include <QProcess>

class ControlsManager : public QObject {
  Q_OBJECT

private:
  EngineController m_engineController;
  JoysticksController *m_manualController;
  QThread *m_manualControllerThread;
  DrivingMode m_currentMode;
  QThread* m_sharedMemoryThread;
  QThread* m_processMonitorThread;
  std::atomic<bool> m_threadRunning;

public:
  /*!
   * Constructs a ControlsManager instance, initializing the engine controller,
   * joystick controller, and setting up the necessary connections and threads.
   *
   * This constructor sets up the EngineController signals to be connected to
   * the ControlsManager signals. It initializes the JoysticksController with
   * callbacks for steering and speed. It also initializes and starts the
   * joystick controller in a separate thread.
   *
   * @param parent The parent QObject for this instance.
   */
  explicit ControlsManager(QObject *parent = nullptr);

  /*!
   * Destroys the ControlsManager instance, ensuring that the joystick
   * controller and its thread are properly cleaned up.
   *
   * This destructor stops the joystick controller, waits for the thread to
   * finish, and deletes the JoysticksController object.
   */
  ~ControlsManager();

  /*!
   * Sets the current driving mode.
   *
   * This function updates the `m_currentMode` member variable. If the mode has
   * not changed, the function returns early. It is called to switch between
   * Manual and Automatic modes.
   *
   * @param mode The new driving mode to set.
   */
  void setMode(DrivingMode mode);

  void  readSharedMemory();

  //bool isServiceRunning(const QString &serviceName);

  bool isProcessRunning(const QString &processName);

signals:
  /*!
   * Signal emitted when the direction of the car is updated.
   *
   * @param newDirection The new direction of the car.
   */
  void directionChanged(CarDirection newDirection);

  /*!
   * Signal emitted when the steering angle of the car is updated.
   *
   * @param newAngle The new steering angle.
   */
  void steeringChanged(int newAngle);

private:
  int shm_fd;
  void* ptr;
};

#endif // CONTROLSMANAGER_HPP
