/**
 * @file MCP2515Controller.hpp
 * @author Michel Batista (michel_fab@outlook.com)
 * @brief
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MCP2515CONTROLLER_HPP
#define MCP2515CONTROLLER_HPP

#include "CANMessageProcessor.hpp"
#include "IMCP2515Controller.hpp"
#include "ISPIController.hpp"
#include "MCP2515Configurator.hpp"
#include <QObject>
#include <string>

class MCP2515Controller : public IMCP2515Controller {
  Q_OBJECT
public:
  explicit MCP2515Controller(const std::string &spiDevice);
  MCP2515Controller(const std::string &spiDevice,
                    ISPIController &spiController);

  ~MCP2515Controller() override;

  bool init() override;
  void processReading() override;
  void stopReading() override;

  CANMessageProcessor &getMessageProcessor() { return messageProcessor; }
  bool isStopReadingFlagSet() const override;

signals:
  void speedUpdated(float newSpeed);
  void rpmUpdated(int newRpm);

private:
  ISPIController *spiController;
  MCP2515Configurator configurator;
  CANMessageProcessor messageProcessor;
  bool stopReadingFlag = false;
  bool ownsSPIController = false;

  void setupHandlers();
};

#endif // MCP2515CONTROLLER_HPP
