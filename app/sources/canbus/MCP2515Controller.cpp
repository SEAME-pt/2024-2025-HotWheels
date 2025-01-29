#include "MCP2515Controller.hpp"
#include <QDebug>
#include <QThread>
#include "SPIController.hpp"
#include <cstring>
#include <stdexcept>

MCP2515Controller::MCP2515Controller(const std::string &spiDevice)
    : spiController(new SPIController())
    , configurator(*spiController)
    , messageProcessor()
    , ownsSPIController(true)
{
    if (!spiController->openDevice(spiDevice)) {
        throw std::runtime_error("Failed to open SPI device : " + spiDevice);
    }
    setupHandlers();
}

MCP2515Controller::MCP2515Controller(const std::string &spiDevice, ISPIController &spiController)
    : spiController(&spiController)
    , configurator(spiController)
    , messageProcessor()
    , ownsSPIController(false)
{
    if (!spiController.openDevice(spiDevice)) {
        throw std::runtime_error("Failed to open SPI device : " + spiDevice);
    }
    setupHandlers();
}

MCP2515Controller::~MCP2515Controller()
{
    spiController->closeDevice();
    if (this->ownsSPIController) {
        delete this->spiController;
    }
}

bool MCP2515Controller::init()
{
    if (!configurator.resetChip()) {
        throw std::runtime_error("Failed to reset MCP2515");
    }

    configurator.configureBaudRate();
    configurator.configureTXBuffer();
    configurator.configureRXBuffer();
    configurator.configureFiltersAndMasks();
    configurator.configureInterrupts();
    configurator.setMode(0x00);

    if (!configurator.verifyMode(0x00)) {
        throw std::runtime_error("Failed to set MCP2515 to normal mode");
    }

    return true;
}

void MCP2515Controller::processReading()
{
    stopReadingFlag = false;
    while (!stopReadingFlag) {
        uint16_t frameID;
        std::vector<uint8_t> data;

        try {
            data = configurator.readCANMessage(frameID);
            if (!data.empty()) {
                messageProcessor.processMessage(frameID, data);
            }
        } catch (const std::exception &e) {
            qDebug() << "Error while processing CAN message:" << e.what();
        }

        QThread::msleep(10);
    }
}

void MCP2515Controller::stopReading()
{
    stopReadingFlag = true;
}

void MCP2515Controller::setupHandlers()
{
    messageProcessor.registerHandler(0x100, [this](const std::vector<uint8_t> &data) {
        if (data.size() == sizeof(float)) {
            float speed;
            memcpy(&speed, data.data(), sizeof(float));
            emit speedUpdated(speed / 10.0f);
        }
    });

    messageProcessor.registerHandler(0x200, [this](const std::vector<uint8_t> &data) {
        if (data.size() == 2) {
            uint16_t rpm = (data[0] << 8) | data[1];
            emit rpmUpdated(rpm);
        }
    });
}

bool MCP2515Controller::isStopReadingFlagSet() const
{
    return this->stopReadingFlag;
}
