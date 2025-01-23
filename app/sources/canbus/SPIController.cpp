#include "SPIController.hpp"
#include <cstring>
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

SPIController::SPIController()
    : spi_fd(-1)
    , mode(DefaultMode)
    , bits(DefaultBitsPerWord)
    , speed(DefaultSpeedHz)
{}

SPIController::~SPIController()
{
    closeDevice();
}

bool SPIController::openDevice(const std::string &device)
{
    spi_fd = open(device.c_str(), O_RDWR);
    if (spi_fd < 0) {
        perror("Failed to open SPI device");
        return false;
    }
    return true;
}

void SPIController::configure(uint8_t mode, uint8_t bits, uint32_t speed)
{
    if (spi_fd < 0) {
        throw std::runtime_error("SPI device not open");
    }

    this->mode = mode;
    this->bits = bits;
    this->speed = speed;

    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) < 0) {
        throw std::runtime_error("Failed to set SPI mode");
    }

    if (ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
        throw std::runtime_error("Failed to set SPI bits per word");
    }

    if (ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        throw std::runtime_error("Failed to set SPI speed");
    }
}

void SPIController::writeByte(uint8_t address, uint8_t data)
{
    uint8_t tx[] = {static_cast<uint8_t>(Opcode::Write), address, data};
    spiTransfer(tx, nullptr, sizeof(tx));
}

uint8_t SPIController::readByte(uint8_t address)
{
    uint8_t tx[] = {static_cast<uint8_t>(Opcode::Read), address, 0x00};
    uint8_t rx[sizeof(tx)] = {0};
    spiTransfer(tx, rx, sizeof(tx));
    return rx[2];
}

void SPIController::spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length)
{
    if (spi_fd < 0) {
        throw std::runtime_error("SPI device not open");
    }

    struct spi_ioc_transfer transfer = {};
    transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
    transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
    transfer.len = length;
    transfer.speed_hz = speed;
    transfer.bits_per_word = bits;

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &transfer) < 0) {
        perror("SPI transfer failed");
        throw std::runtime_error("SPI transfer error");
    }
}

void SPIController::closeDevice()
{
    if (spi_fd >= 0) {
        close(spi_fd);
        spi_fd = -1;
    }
}
