#ifndef MCP2515_HPP
#define MCP2515_HPP

#include <QObject>
#include <cstdint>
#include <string>
#include <vector>

class MCP2515 : public QObject
{
    Q_OBJECT
private:
    int spi_fd;
    bool disabled = false;

    void disableSpi();
    bool isDisabled();

    // SPI Configuration Constants
    static constexpr uint32_t SPI_SPEED = 1000000; // 1 MHz
    static constexpr uint8_t SPI_BITS_PER_WORD = 8;
    static constexpr uint8_t SPI_MODE = 0;

    // MCP2515 Commands
    static constexpr uint8_t CAN_RESET = 0xC0;
    static constexpr uint8_t CAN_WRITE = 0x02;
    static constexpr uint8_t CAN_READ = 0x03;

    // MCP2515 Register Addresses
    static constexpr uint8_t CNF1 = 0x2A;
    static constexpr uint8_t CNF2 = 0x29;
    static constexpr uint8_t CNF3 = 0x28;
    static constexpr uint8_t CANCTRL = 0x0F;
    static constexpr uint8_t CANSTAT = 0x0E;

    static constexpr uint8_t CANINTF = 0x2C;
    static constexpr uint8_t CANINTE = 0x2B;
    static constexpr uint8_t RXB0SIDH = 0x61;
    static constexpr uint8_t RXB0SIDL = 0x62;
    static constexpr uint8_t RXB0DLC = 0x65;

    // Configurable Parameters
    static constexpr uint8_t BAUD_RATE_CNF1 = 0x00;           // Set baud rate (500 kbps)
    static constexpr uint8_t CNF2_VALUE = 0x80 | 0x10 | 0x00; // PHSEG1_3TQ | PRSEG_1TQ
    static constexpr uint8_t CNF3_VALUE = 0x02;               // PHSEG2_3TQ
    static constexpr uint8_t NORMAL_MODE = 0x00;

    void configureSPI();
    void configureBaudRate();
    void configureMode(uint8_t mode);
    void verifyMode(uint8_t expectedMode);
    void writeByte(uint8_t address, uint8_t data);
    uint8_t readByte(uint8_t address);
    void sendCommand(uint8_t command);
    void spiTransfer(const uint8_t *tx, size_t length);
    void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length);
    void handleData(const std::vector<uint8_t> &CAN_RX_Buf);

public:
    explicit MCP2515(const std::string &spi_device);
    ~MCP2515();

    void reset();
    bool init();
    std::vector<uint8_t> receive();

signals:
    void speedUpdated(int newSpeed);
    void rpmUpdated(int newRpm);
};

#endif // MCP2515_HPP
