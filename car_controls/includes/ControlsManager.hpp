#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "../middleware/ClientThread.hpp"
#include "../middleware/CarDataI.hpp"
#include <QObject>
#include <QThread>
#include <QProcess>

class ControlsManager : public QObject {
  Q_OBJECT

private:
  EngineController m_engineController;
  JoysticksController *m_manualController;
  DrivingMode m_currentMode;
  ClientThread *m_clientObject;
  Data::CarDataI *m_carDataObject;

  QThread *m_manualControllerThread;
  QThread* m_processMonitorThread;
  QThread* m_carDataThread;
  QThread* m_clientThread;

  std::atomic<bool> m_threadRunning;

public:
  explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
  ~ControlsManager();

  void setMode(DrivingMode mode);
  void readJoystickEnable();
  bool isProcessRunning(const QString &processName);
  //bool isServiceRunning(const QString &serviceName);
};

#endif // CONTROLSMANAGER_HPP
