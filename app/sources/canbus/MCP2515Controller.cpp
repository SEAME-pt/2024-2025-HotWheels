#include "MCP2515Controller.hpp"
#include <QDebug>
#include <QThread>
#include <cstring>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

MCP2515Controller::MCP2515Controller(const std::string &spi_device)
{
    spi_fd = open(spi_device.c_str(), O_RDWR);
    if (spi_fd < 0) {
        perror("Failed to open SPI device");
        disabled = true;
    } else {
        configureSPI();
    }
}

MCP2515Controller::~MCP2515Controller()
{
    if (!disabled) {
        close(spi_fd);
    }
}

void MCP2515Controller::configureSPI()
{
    uint8_t mode = 0, bits = 8;
    uint32_t speed = 1000000;

    ioctl(spi_fd, SPI_IOC_WR_MODE, &mode);
    ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
}

bool MCP2515Controller::init()
{
    if (disabled)
        return false;

    sendCommand(0xC0);
    usleep(100000);

    uint8_t status = readByte(0x0E);
    if ((status & 0xE0) != 0x80) {
        qDebug() << "Reset failed: MCP2515 not in configuration mode.";
        return false;
    }

    configureBaudRate();
    configureTXBuffer();
    configureRXBuffer();
    configureFiltersAndMasks();
    configureInterrupts();

    configureMode(0x00);
    verifyMode(0x00);

    return true;
}

void MCP2515Controller::configureBaudRate()
{
    writeByte(0x2A, 0x00);
    writeByte(0x29, 0x80 | 0x10 | 0x00);
    writeByte(0x28, 0x02);
}

void MCP2515Controller::configureTXBuffer()
{
    writeByte(0x31, 0xFF);
    writeByte(0x32, 0xE0);
    writeByte(0x35, 0x40 | 0x08);
}

void MCP2515Controller::configureRXBuffer()
{
    writeByte(0x61, 0x00);
    writeByte(0x62, 0x60);
    writeByte(0x60, 0x60);
    writeByte(0x65, 0x08);
}

void MCP2515Controller::configureFiltersAndMasks()
{
    writeByte(0x00, 0xFF);
    writeByte(0x01, 0xE0);
    writeByte(0x20, 0xFF);
    writeByte(0x21, 0xE0);
}

void MCP2515Controller::configureInterrupts()
{
    writeByte(0x2C, 0x00);
    writeByte(0x2B, 0x01);
}

void MCP2515Controller::configureMode(uint8_t mode)
{
    writeByte(0x0F, mode | 0x04);
}

void MCP2515Controller::verifyMode(uint8_t expectedMode)
{
    uint8_t mode = readByte(0x0E);
    if ((mode & 0xE0) != expectedMode) {
        qDebug() << "Mode verification failed. Retrying...";
        configureMode(expectedMode | 0x04);
    }
}

void MCP2515Controller::writeByte(uint8_t address, uint8_t data)
{
    uint8_t tx[] = {0x02, address, data};
    spiTransfer(tx, sizeof(tx));
}

uint8_t MCP2515Controller::readByte(uint8_t address)
{
    uint8_t tx[] = {0x03, address, 0x00};
    uint8_t rx[sizeof(tx)] = {0};
    spiTransfer(tx, rx, sizeof(tx));
    return rx[2];
}

void MCP2515Controller::sendCommand(uint8_t command)
{
    uint8_t tx[] = {command};
    spiTransfer(tx, sizeof(tx));
}

void MCP2515Controller::spiTransfer(const uint8_t *tx, size_t length)
{
    std::vector<uint8_t> rx(length, 0);
    spiTransfer(tx, rx.data(), length);
}

void MCP2515Controller::spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length)
{
    if (disabled)
        return;

    struct spi_ioc_transfer transfer = {};
    transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
    transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
    transfer.len = length;
    transfer.speed_hz = 1000000;
    transfer.bits_per_word = 8;

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
        perror("SPI transfer failed");
        throw std::runtime_error("SPI transfer error");
    }
}

void MCP2515Controller::processReading()
{
    while (!m_stopReading) {
        uint16_t frameID;
        std::vector<uint8_t> CAN_RX_Buf = readCANData(frameID);

        if (!CAN_RX_Buf.empty()) {
            processCANMessage(frameID, CAN_RX_Buf);
        }

        resetRxBuffer();

        QThread::msleep(10);
    }
}

std::vector<uint8_t> MCP2515Controller::readCANData(uint16_t &frameID)
{
    std::vector<uint8_t> CAN_RX_Buf;

    if (readByte(0x2C) & 0x01) {
        uint8_t sidh = readByte(0x61);
        uint8_t sidl = readByte(0x62);
        frameID = (sidh << 3) | (sidl >> 5);

        uint8_t len = readByte(0x65);
        for (uint8_t i = 0; i < len; ++i) {
            CAN_RX_Buf.push_back(readByte(0x66 + i));
        }
    }

    return CAN_RX_Buf;
}

void MCP2515Controller::processCANMessage(uint16_t frameID, const std::vector<uint8_t> &CAN_RX_Buf)
{
    if (frameID == 0x100) {
        handleSpeedData(CAN_RX_Buf);
    } else if (frameID == 0x200) {
        handleRPMData(CAN_RX_Buf);
    } else {
        qDebug() << "Unknown frameId in CANBUS Data" << frameID;
    }
}

void MCP2515Controller::handleSpeedData(const std::vector<uint8_t> &data)
{
    if (data.size() == sizeof(float)) {
        float speed;
        std::memcpy(&speed, data.data(), sizeof(float));
        // qDebug() << "CAN Data SPEED received" << speed;
        emit speedUpdated(speed / 10.0f);
    }
}

void MCP2515Controller::handleRPMData(const std::vector<uint8_t> &data)
{
    if (data.size() == 2) {
        uint16_t rpm = (data[0] << 8) | data[1];
        // qDebug() << "CAN Data RPM received" << rpm;
        emit rpmUpdated(rpm);
    }
}

void MCP2515Controller::resetRxBuffer()
{
    writeByte(CANINTF, 0x00);  // Clear interrupt flag
    writeByte(CANINTE, 0x01);  // Re-enable RX interrupt
    writeByte(RXB0SIDH, 0x00); // Reset RX buffer
    writeByte(RXB0SIDL, 0x60);
}

void MCP2515Controller::stopReading()
{
    m_stopReading = true;
}
