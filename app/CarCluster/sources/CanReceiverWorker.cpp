#include "CanReceiverWorker.hpp"
#include <QDebug>
#include <QThread>

CanReceiverWorker::CanReceiverWorker(MCP2515 *canBusController, QObject *parent)
    : QObject(parent)
    , m_canBusController(canBusController)
{}

void CanReceiverWorker::process()
{
    while (true) {
        // qDebug() << "Tries receiving";
        m_canBusController->receive(); // Continuously read CAN messages
        QThread::msleep(10);           // Sleep for 10 ms to avoid hogging the CPU
    }
}
