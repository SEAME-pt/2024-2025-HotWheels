#ifndef MOCKMCP2515CONTROLLER_HPP
#define MOCKMCP2515CONTROLLER_HPP

#include <QDebug>
#include "IMCP2515Controller.hpp"
#include <gmock/gmock.h>

class MockMCP2515Controller : public IMCP2515Controller
{
    Q_OBJECT

public:
    MOCK_METHOD(bool, init, (), (override));
    MOCK_METHOD(void, processReading, (), (override));
    MOCK_METHOD(void, stopReading, (), (override));
    MOCK_METHOD(bool, isStopReadingFlagSet, (), (const, override));

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);
};

#endif // MOCKMCP2515CONTROLLER_HPP
