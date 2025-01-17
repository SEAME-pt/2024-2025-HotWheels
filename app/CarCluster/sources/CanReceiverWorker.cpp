#include "CanReceiverWorker.hpp"
#include <QDebug>
#include <QThread>
#include "SystemInfoUtility.hpp"

CanReceiverWorker::CanReceiverWorker(MCP2515 *canBusController, QObject *parent)
    : QObject(parent)
    , m_canBusController(canBusController)
{
    SystemInfoUtility::printClassInfo("CanReceiverWorker", SystemInfoUtility::InfoType::CreBegin);
    SystemInfoUtility::printClassInfo("CanReceiverWorker", SystemInfoUtility::InfoType::CreEnd);
}

CanReceiverWorker::~CanReceiverWorker()
{
    SystemInfoUtility::printClassInfo("CanReceiverWorker", SystemInfoUtility::InfoType::DesBegin);
    SystemInfoUtility::printClassInfo("CanReceiverWorker", SystemInfoUtility::InfoType::DesEnd);
}

void CanReceiverWorker::process()
{
    while (true) {
        qDebug() << "Tries receiving";
        m_canBusController->receive(); // Continuously read CAN messages
        QThread::msleep(10);           // Sleep for 10 ms to avoid hogging the CPU
    }
}
