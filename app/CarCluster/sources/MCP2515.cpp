#include "MCP2515.hpp"
#include <QDateTime>
#include <QDebug>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/spi/spidev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

MCP2515::MCP2515(const std::string &spi_device)
{
    spi_fd = open(spi_device.c_str(), O_RDWR);
    if (spi_fd < 0) {
        perror("Failed to open SPI device");
        throw std::runtime_error("Unable to open SPI device.");
    }
    configureSPI();
}

MCP2515::~MCP2515()
{
    close(spi_fd);
}

void MCP2515::configureSPI()
{
    ioctl(spi_fd, SPI_IOC_WR_MODE, &SPI_MODE);
    ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &SPI_BITS_PER_WORD);
    ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &SPI_SPEED);
}

void MCP2515::reset()
{
    sendCommand(CAN_RESET);
}

void MCP2515::configureBaudRate()
{
    writeByte(CNF1, BAUD_RATE_CNF1);
    writeByte(CNF2, CNF2_VALUE);
    writeByte(CNF3, CNF3_VALUE);
}

void MCP2515::configureMode(uint8_t mode)
{
    writeByte(CANCTRL, mode);
}

void MCP2515::verifyMode(uint8_t expectedMode)
{
    uint8_t mode = readByte(CANSTAT);
    if ((mode & 0xE0) != expectedMode) {
        std::cerr << "Failed to set CAN controller to expected mode." << std::endl;
        configureMode(expectedMode);
    }
}

void MCP2515::writeByte(uint8_t address, uint8_t data)
{
    uint8_t tx[] = {CAN_WRITE, address, data};
    spiTransfer(tx, sizeof(tx));
}

uint8_t MCP2515::readByte(uint8_t address)
{
    uint8_t tx[] = {CAN_READ, address, 0x00};
    uint8_t rx[sizeof(tx)] = {0};
    spiTransfer(tx, rx, sizeof(tx));
    return rx[2];
}

void MCP2515::sendCommand(uint8_t command)
{
    uint8_t tx[] = {command};
    spiTransfer(tx, sizeof(tx));
}

void MCP2515::spiTransfer(const uint8_t *tx, size_t length)
{
    std::vector<uint8_t> rx(length, 0);
    spiTransfer(tx, rx.data(), length);
}

void MCP2515::spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length)
{
    struct spi_ioc_transfer transfer;
    memset(&transfer, 0, sizeof(transfer));
    transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
    transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
    transfer.len = static_cast<__u32>(length); // Explicit narrowing cast
    transfer.speed_hz = SPI_SPEED;
    transfer.bits_per_word = SPI_BITS_PER_WORD;

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
        perror("SPI transfer failed");
        throw std::runtime_error("SPI transfer error");
    }
}

bool MCP2515::init()
{
    std::cout << "Initializing CAN Controller..." << std::endl;

    reset();
    usleep(100000); // Allow time for reset

    configureBaudRate();
    configureMode(NORMAL_MODE);

    verifyMode(NORMAL_MODE);
    std::cout << "CAN Controller Initialized Successfully." << std::endl;

    return true;
}

void MCP2515::handleData(const std::vector<uint8_t> &CAN_RX_Buf)
{
    if (CAN_RX_Buf.empty()) {
        qDebug() << "[WARNING] Received an empty data frame.";
        return;
    }

    if (CAN_RX_Buf.size() == 1) {
        // Speed data (1-byte)
        int speedValue = static_cast<int>(CAN_RX_Buf[0]);
        emit speedUpdated(speedValue);
        // qDebug() << "Received Speed:" << speedValue;
    } else if (CAN_RX_Buf.size() == 2) {
        // RPM data (2-byte)
        uint16_t rpm = (static_cast<uint16_t>(CAN_RX_Buf[0]) << 8)
                       | static_cast<uint16_t>(CAN_RX_Buf[1]);
        emit rpmUpdated(rpm);
        // qDebug() << "Received RPM:" << rpm;
    } else {
        qDebug() << "[WARNING] Received data with an unexpected size:" << CAN_RX_Buf.size();
    }
}

std::vector<uint8_t> MCP2515::receive()
{
    std::vector<uint8_t> CAN_RX_Buf;

    while (true) {
        if (readByte(CANINTF) & 0x01) {      // Check if data is available in the RX buffer
            uint8_t len = readByte(RXB0DLC); // Read the data length code (DLC)
            for (uint8_t i = 0; i < len; ++i) {
                CAN_RX_Buf.push_back(readByte(0x66 + i)); // RXB0D0 is 0x66
            }
            break;
        }
    }

    handleData(CAN_RX_Buf);

    writeByte(CANINTF, 0x00);  // Clear interrupt flag
    writeByte(CANINTE, 0x01);  // Re-enable RX interrupt
    writeByte(RXB0SIDH, 0x00); // Reset RX buffer
    writeByte(RXB0SIDL, 0x60);

    return CAN_RX_Buf;
}
