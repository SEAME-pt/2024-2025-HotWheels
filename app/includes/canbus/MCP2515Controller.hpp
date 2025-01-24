#ifndef MCP2515CONTROLLER_HPP
#define MCP2515CONTROLLER_HPP

#include <QObject>
#include "CANMessageProcessor.hpp"
#include "MCP2515Configurator.hpp"
#include "SPIController.hpp"
#include <string>

class MCP2515Controller : public QObject
{
    Q_OBJECT
public:
    explicit MCP2515Controller(const std::string &spiDevice);
    ~MCP2515Controller();

    bool init();
    void processReading();
    void stopReading();

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);

private:
    SPIController spiController;
    MCP2515Configurator configurator;
    CANMessageProcessor messageProcessor;
    bool stopReadingFlag = false;

    void setupHandlers();
};

#endif // MCP2515CONTROLLER_HPP
