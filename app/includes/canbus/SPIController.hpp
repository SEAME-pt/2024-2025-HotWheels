#ifndef SPICONTROLLER_HPP
#define SPICONTROLLER_HPP

#include "ISPIController.hpp"
#include <string>
#include <cstdint>

class SPIController : public ISPIController {
public:
    SPIController();
    ~SPIController();

    bool openDevice(const std::string& device) override;
    void configure(uint8_t mode, uint8_t bits, uint32_t speed) override;
    void writeByte(uint8_t address, uint8_t data) override;
    uint8_t readByte(uint8_t address) override;
    void spiTransfer(const uint8_t* tx, uint8_t* rx, size_t length) override;
    void closeDevice() override;

private:
    int spi_fd;
    uint8_t mode;
    uint8_t bits;
    uint32_t speed;
};

#endif // SPICONTROLLER_HPP
