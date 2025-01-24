#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include <cstdint>
#include <string>

class SPIController
{
public:
    enum class Opcode : uint8_t { Write = 0x02, Read = 0x03 };

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

    static constexpr uint8_t DefaultBitsPerWord = 8;
    static constexpr uint32_t DefaultSpeedHz = 1'000'000;
    static constexpr uint8_t DefaultMode = 0;
};

#endif // SPICONTROLLER_HPP
