#ifndef IMCP2515CONTROLLER_HPP
#define IMCP2515CONTROLLER_HPP

#include <QObject>

class IMCP2515Controller : public QObject
{
    Q_OBJECT
public:
    virtual ~IMCP2515Controller() = default;

    virtual bool init() = 0;
    virtual void processReading() = 0;
    virtual void stopReading() = 0;

    virtual bool isStopReadingFlagSet() const = 0;

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);
};

#endif // IMCP2515CONTROLLER_HPP
