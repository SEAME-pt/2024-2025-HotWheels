#ifndef MCP2515CONTROLLER_HPP
#define MCP2515CONTROLLER_HPP

#include <QObject>
#include <cstdint>
#include <string>
#include <vector>

class MCP2515Controller : public QObject
{
    Q_OBJECT
public:
    explicit MCP2515Controller(const std::string &spi_device);
    ~MCP2515Controller();

    bool init();
    void processReading();
    void stopReading();

signals:
    void speedUpdated(float newSpeed);
    void rpmUpdated(int newRpm);

private:
    int spi_fd;
    bool disabled = false;
    bool m_stopReading = false;

    void configureSPI();
    void configureBaudRate();
    void configureMode(uint8_t mode);
    void verifyMode(uint8_t expectedMode);
    void configureTXBuffer();
    void configureTXBuffer1();
    void configureRXBuffer();
    void configureFiltersAndMasks();
    void configureInterrupts();
    void writeByte(uint8_t address, uint8_t data);
    uint8_t readByte(uint8_t address);
    void sendCommand(uint8_t command);
    void spiTransfer(const uint8_t *tx, size_t length);
    void spiTransfer(const uint8_t *tx, uint8_t *rx, size_t length);
    void processCANMessage(uint16_t frameID, const std::vector<uint8_t> &CAN_RX_Buf);
    void handleSpeedData(const std::vector<uint8_t> &data);
    void handleRPMData(const std::vector<uint8_t> &data);
    std::vector<uint8_t> readCANData(uint16_t &frameID);
    void resetRxBuffer();

    static constexpr uint8_t CANINTF = 0x2C;
    static constexpr uint8_t CANINTE = 0x2B;
    static constexpr uint8_t RXB0SIDH = 0x61;
    static constexpr uint8_t RXB0SIDL = 0x62;
};

#endif // MCP2515CONTROLLER_HPP
