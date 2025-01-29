#ifndef CANBUSMANAGER_HPP
#define CANBUSMANAGER_HPP

#include <QObject>
#include <QThread>
#include "IMCP2515Controller.hpp"

class CanBusManager : public QObject
{
    Q_OBJECT
public:
    explicit CanBusManager(const std::string &spi_device, QObject *parent = nullptr);
    CanBusManager(IMCP2515Controller *controller, QObject *parent = nullptr);

    ~CanBusManager();
    bool initialize();

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);

private:
    IMCP2515Controller *m_controller = nullptr;
    QThread *m_thread = nullptr;
    bool ownsMCP2515Controller = false;

    void connectSignals();
};

#endif // CANBUSMANAGER_HPP
