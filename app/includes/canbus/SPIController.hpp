#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include <string>
#include <cstdint>

class SPIController
{
public:
    SPIController();
    ~SPIController();

    bool openDevice(const std::string &device);
    void configure(uint8_t mode, uint8_t bits, uint32_t speed);
    void writeByte(uint8_t address, uint8_t data);
    uint8_t readByte(uint8_t address);
    void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length);
    void closeDevice();

private:
    int spi_fd;
    uint8_t mode;
    uint8_t bits;
    uint32_t speed;
};

#endif // SPICONTROLLER_HPP
