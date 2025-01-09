#include "CanController.hpp"
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QTimer>

CanController::CanController(QObject *parent)
    : QObject(parent)
    , canDevice(nullptr)
{}

CanController::~CanController()
{
    if (canDevice) {
        canDevice->disconnectDevice();
        delete canDevice;
    }
}

bool CanController::connectDevice()
{
    return connectCan0();
}

bool CanController::connectCan0()
{
    canDevice = QCanBus::instance()->createDevice("socketcan", "can0", nullptr);
    if (!canDevice) {
        qDebug() << "Failed to create CAN device: Plugin not loaded or invalid parameters.";
        return false;
    }

    connect(canDevice, &QCanBusDevice::framesReceived, this, &CanController::processReceivedFrames);

    if (canDevice->connectDevice()) {
        qDebug() << "Connected to CAN interface: can0";
        return true;
    } else {
        qDebug() << "Failed to connect to CAN interface:" << canDevice->errorString();
        delete canDevice;
        canDevice = nullptr;
    }
    return false;
}

void CanController::processReceivedFrames()
{
    // qDebug() << "FrameSize" << canDevice->framesAvailable();
    while (canDevice->framesAvailable()) {
        QCanBusFrame frame = canDevice->readFrame();

        // Get the current timestamp
        QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz");

        qDebug() << "[" << timestamp << "]"
                 << "Received CAN Frame:"
                 << "ID:" << frame.frameId() << "Data:" << frame.payload().toHex()
                 << "Data size:" << frame.payload().size();

        if (frame.frameId() == 0x100) {
            // QString speedData = QString::fromUtf8(frame.payload().toHex());
            QString speedData = frame.payload().toHex();
            int speedValue = speedData.toInt(nullptr, 16);
            emit speedUpdated(speedValue);
        } else if (frame.frameId() == 0x200) {
            const QByteArray payload = frame.payload();

            // Ensure we have at least 2 bytes
            // qDebug() << "Frame content" << payload.isLower();
            if (payload.size() >= 2) {
                // Reconstruct RPM from 2 bytes (high byte first)
                uint16_t rpm = (static_cast<uint16_t>(payload[0] & 0xFF) << 8)
                               | (static_cast<uint16_t>(payload[1] & 0xFF));

                emit rpmUpdated(rpm);

                // Optional: Print for debugging
                qDebug() << "Received RPM:" << rpm;
            } else {
                // qDebug() << "Not enough bytes" << payload;
            }
        } else {
            qDebug() << "[" << timestamp << "]"
                     << "[WARNING] Invalid data received.";
        }
    }
}
