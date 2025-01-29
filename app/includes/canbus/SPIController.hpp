#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include "ISPIController.hpp"
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

using IoctlFunc = int (*)(int, unsigned long, ...);
using OpenFunc = int (*)(const char *, int, ...);
using CloseFunc = int (*)(int);

class SPIController : public ISPIController
{
public:
    enum class Opcode : uint8_t { Write = 0x02, Read = 0x03 };

    SPIController(IoctlFunc ioctlFunc = ::ioctl,
                  OpenFunc openFunc = ::open,
                  CloseFunc closeFunc = ::close);
    ~SPIController() override;

    bool openDevice(const std::string &device) override;
    void configure(uint8_t mode, uint8_t bits, uint32_t speed) override;
    void writeByte(uint8_t address, uint8_t data) override;
    uint8_t readByte(uint8_t address) override;
    void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length) override;
    void closeDevice() override;

private:
    int spi_fd;
    uint8_t mode;
    uint8_t bits;
    uint32_t speed;

    IoctlFunc m_ioctlFunc;
    OpenFunc m_openFunc;
    CloseFunc m_closeFunc;

    static constexpr uint8_t DefaultBitsPerWord = 8;
    static constexpr uint32_t DefaultSpeedHz = 1'000'000;
    static constexpr uint8_t DefaultMode = 0;
};

#endif // SPICONTROLLER_HPP
