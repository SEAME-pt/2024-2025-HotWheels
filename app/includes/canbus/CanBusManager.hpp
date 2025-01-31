/**
 * @file CanBusManager.hpp
 * @author Michel Batista (michel_fab@outlook.com)
 * @brief
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include "IMCP2515Controller.hpp"
#include <QObject>
#include <QThread>

/**
 * @brief Class that manages the CAN bus communication.
 * @class CanBusManager inherits from QObject
 */
class CanBusManager : public QObject {
  /** @brief Macro required for Qt's meta-object system. */
  Q_OBJECT
public:
  /** @brief Constructor for the CanBusManager class that takes a SPI device
   * path. */
  explicit CanBusManager(const std::string &spi_device,
                         QObject *parent = nullptr);
  /** @brief Constructor for the CanBusManager class that takes a MCP2515
   * controller. */
  CanBusManager(IMCP2515Controller *controller, QObject *parent = nullptr);
  /** @brief Destructor for the CanBusManager class. */
  ~CanBusManager();
  /** @brief Method to initialize the CanBusManager. */
  bool initialize();

signals:
  /** @brief Signal emitted when the speed is updated. */
  void speedUpdated(float newSpeed);
  /** @brief Signal emitted when the RPM is updated. */
  void rpmUpdated(int newRpm);

private:
  /** @brief Pointer to the IMCP2515Controller object. */
  IMCP2515Controller *m_controller = nullptr;
  /** @brief Pointer to the QThread object. */
  QThread *m_thread = nullptr;
  /** @brief Flag to indicate if the MCP2515 controller is owned by the
   * CanBusManager. */
  bool ownsMCP2515Controller = false;
  /** @brief Method to connect signals. */
  void connectSignals();
};

#endif // CANBUSMANAGER_HPP
