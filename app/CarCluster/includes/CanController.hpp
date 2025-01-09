#ifndef CANCONTROLLER_HPP
#define CANCONTROLLER_HPP

#include <QCanBus>
#include <QCanBusDevice>
#include <QObject>

class CanController : public QObject
{
    Q_OBJECT

public:
    explicit CanController(QObject *parent = nullptr);
    ~CanController();

    bool connectDevice();

signals:
    void speedUpdated(int newSpeed);
    void rpmUpdated(int newRpm);

private slots:
    void processReceivedFrames();

private:
    bool connectCan0();

private:
    QCanBusDevice *canDevice;
};

#endif // CANCONTROLLER_HPP
