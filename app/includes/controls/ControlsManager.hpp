/*!
 * @file ControlsManager.hpp
 * @brief Definition of the ControlsManager class.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the definition of the ControlsManager class,
 * which is responsible for managing the car's controls.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "enums.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include "../ZeroC/ClientThread.hpp"
#include <QObject>
#include <QThread>

/*!
 * @brief Class that manages the car's controls.
 * @class ControlsManager inherits from QObject
 */
class ControlsManager : public QObject {
  Q_OBJECT

public:
  explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
  ~ControlsManager();

public slots:
  void drivingModeUpdated(DrivingMode newMode);

signals:
  void directionChanged(CarDirection newDirection);
  void steeringChanged(int newAngle);

private:
  ClientThread *m_clientObject;

  QThread* m_clientThread;
};

#endif // CONTROLSMANAGER_HPP
