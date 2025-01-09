#include "SpiController.hpp"
#include <QDebug>
#include <errno.h>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

// MCP2515 SPI Commands
#define MCP2515_RESET 0xC0
#define MCP2515_READ 0x03
#define MCP2515_READ_RX 0x90
#define MCP2515_READ_STATUS 0xA0
#define MCP2515_RX_STATUS 0xB0
#define MCP2515_WRITE 0x02

// MCP2515 Registers
#define MCP2515_RX_CTRL 0x60
#define MCP2515_CANSTAT 0x0E
#define MCP2515_CANCTRL 0x0F
#define MCP2515_RXB0CTRL 0x60
#define MCP2515_RXB0SIDH 0x61
#define MCP2515_RXB0D0 0x66

SpiController::SpiController(QObject *parent)
    : QObject(parent)
    , spifd(-1)
    , devicePath("/dev/spidev1.1")
    , pollTimer(new QTimer(this))
{
    connect(pollTimer, &QTimer::timeout, this, &SpiController::pollMessages);
}

SpiController::~SpiController()
{
    if (spifd >= 0) {
        close(spifd);
    }
    if (pollTimer->isActive()) {
        pollTimer->stop();
    }
}

bool SpiController::connectDevice()
{
    // Open SPI device
    spifd = open(devicePath.toStdString().c_str(), O_RDWR);
    if (spifd < 0) {
        qDebug() << "Failed to open SPI device:" << strerror(errno);
        return false;
    }

    // Set SPI mode (Mode 0)
    uint8_t mode = SPI_MODE_0;
    if (ioctl(spifd, SPI_IOC_WR_MODE, &mode) < 0) {
        qDebug() << "Failed to set SPI mode:" << strerror(errno);
        close(spifd);
        spifd = -1;
        return false;
    }

    // Set bits per word
    uint8_t bits = 8;
    if (ioctl(spifd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
        qDebug() << "Failed to set bits per word:" << strerror(errno);
        close(spifd);
        spifd = -1;
        return false;
    }

    // Set max speed (10MHz - MCP2515 supports up to 10MHz)
    uint32_t speed = 500000;
    if (ioctl(spifd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        qDebug() << "Failed to set speed:" << strerror(errno);
        close(spifd);
        spifd = -1;
        return false;
    }

    // Reset MCP2515
    if (!reset()) {
        qDebug() << "Failed to reset MCP2515";
        close(spifd);
        spifd = -1;
        return false;
    }

    pollTimer->start(10);

    qDebug() << "Connected to MCP2515 via SPI";
    return true;
}

bool SpiController::reset()
{
    uint8_t cmd = MCP2515_RESET;
    return writeSPI(&cmd, 1);
}

bool SpiController::writeSPI(const uint8_t *data, size_t length)
{
    struct spi_ioc_transfer tr = {0};
    tr.tx_buf = (unsigned long) data;
    tr.len = length;
    tr.delay_usecs = 0;
    tr.speed_hz = 500000;
    tr.bits_per_word = 8;

    int ret = ioctl(spifd, SPI_IOC_MESSAGE(1), &tr);
    return (ret == length);
}

bool SpiController::readSPI(uint8_t *data, size_t length)
{
    struct spi_ioc_transfer tr = {0};
    tr.rx_buf = (unsigned long) data;
    tr.len = length;
    tr.delay_usecs = 0;
    tr.speed_hz = 500000;
    tr.bits_per_word = 8;

    int ret = ioctl(spifd, SPI_IOC_MESSAGE(1), &tr);
    return (ret == length);
}

uint8_t SpiController::readRegister(uint8_t address)
{
    uint8_t data[3] = {MCP2515_READ, address, 0};
    struct spi_ioc_transfer tr = {0};
    tr.tx_buf = (unsigned long) data;
    tr.rx_buf = (unsigned long) data;
    tr.len = 3;
    tr.delay_usecs = 0;
    tr.speed_hz = 500000;
    tr.bits_per_word = 8;

    if (ioctl(spifd, SPI_IOC_MESSAGE(1), &tr) < 0) {
        return 0;
    }
    return data[2];
}

bool SpiController::writeRegister(uint8_t address, uint8_t value)
{
    uint8_t data[3] = {MCP2515_WRITE, address, value};
    return writeSPI(data, 3);
}

bool SpiController::readMessage(CanMessage &msg)
{
    // Check if message is available
    uint8_t status = readRegister(MCP2515_CANSTAT);
    if (!(status & 0x01)) { // No message available
        return false;
    }

    // Read message
    uint8_t cmd[14] = {MCP2515_READ_RX}; // Command to read RX buffer 0
    struct spi_ioc_transfer tr = {0};
    tr.tx_buf = (unsigned long) cmd;
    tr.rx_buf = (unsigned long) cmd;
    tr.len = 14;
    tr.delay_usecs = 0;
    tr.speed_hz = 500000;
    tr.bits_per_word = 8;

    if (ioctl(spifd, SPI_IOC_MESSAGE(1), &tr) < 0) {
        return false;
    }

    // Parse the message
    msg.id = (cmd[1] << 3) | (cmd[2] >> 5); // Standard ID
    msg.dlc = cmd[5] & 0x0F;                // Data Length Code

    // Copy data bytes
    for (int i = 0; i < msg.dlc && i < 8; i++) {
        msg.data[i] = cmd[6 + i];
    }

    emit messageReceived(msg);
    return true;
}

void SpiController::pollMessages()
{
    CanMessage msg;
    while (readMessage(msg)) {
        // Message processed in readMessage (emits signal)
    }
}
