#ifndef CANRECEIVERWORKER_HPP
#define CANRECEIVERWORKER_HPP

#include <QObject>
#include "MCP2515.hpp" // Include your MCP2515 class

class CanReceiverWorker : public QObject
{
    Q_OBJECT

public:
    explicit CanReceiverWorker(MCP2515 *canBusController, QObject *parent = nullptr);

public slots:
    void process(); // Slot to process CAN messages in a loop

private:
    MCP2515 *m_canBusController; // Pointer to the MCP2515 instance
};

#endif // CANRECEIVERWORKER_HPP
