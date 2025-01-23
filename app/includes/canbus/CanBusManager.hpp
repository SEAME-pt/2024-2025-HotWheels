#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include <QObject>
#include <QThread>
#include "MCP2515Controller.hpp"
#include "SPIController.hpp"

class CanBusManager : public QObject
{
    Q_OBJECT
public:
    explicit CanBusManager(const std::string &spi_device, QObject *parent = nullptr);
    ~CanBusManager();

    bool initialize();

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);

private:
    SPIController *m_spiController = nullptr;
    MCP2515Controller *m_controller = nullptr;
    QThread *m_thread = nullptr;
};

#endif // CANBUSMANAGER_HPP
