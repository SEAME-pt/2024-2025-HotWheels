#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include <QObject>
#include <QString>
#include <QTimer>

struct CanMessage
{
    uint32_t id;     // CAN ID
    uint8_t dlc;     // Data Length Code
    uint8_t data[8]; // Data bytes
};

class SpiController : public QObject
{
    Q_OBJECT

public:
    explicit SpiController(QObject *parent = nullptr);
    ~SpiController();

    bool connectDevice();
    void pollMessages();

signals:
    void messageReceived(const CanMessage &msg);

private:
    bool reset();
    bool writeSPI(const uint8_t *data, size_t length);
    bool readSPI(uint8_t *data, size_t length);
    uint8_t readRegister(uint8_t address);
    bool writeRegister(uint8_t address, uint8_t value);
    bool readMessage(CanMessage &msg);

    int spifd;
    QString devicePath;
    QTimer *pollTimer;
};

#endif // SPICONTROLLER_HPP
