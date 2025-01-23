#ifndef MCP2515CONFIGURATOR_HPP
#define MCP2515CONFIGURATOR_HPP

#include "SPIController.hpp"
#include <cstdint>
#include <vector>

class MCP2515Configurator {
public:
    explicit MCP2515Configurator(SPIController &spiController);
    ~MCP2515Configurator() = default;

    bool resetChip();
    void configureBaudRate();
    void configureTXBuffer();
    void configureRXBuffer();
    void configureFiltersAndMasks();
    void configureInterrupts();
    void setMode(uint8_t mode);
    bool verifyMode(uint8_t expectedMode);
	std::vector<uint8_t> readCANMessage(uint16_t& frameID);

private:
    SPIController &spiController;

    void writeRegister(uint8_t address, uint8_t value);
    uint8_t readRegister(uint8_t address);
    void sendCommand(uint8_t command);

    static constexpr uint8_t RESET_CMD = 0xC0;
    static constexpr uint8_t CANCTRL = 0x0F;
    static constexpr uint8_t CANSTAT = 0x0E;
    static constexpr uint8_t CNF1 = 0x2A;
    static constexpr uint8_t CNF2 = 0x29;
    static constexpr uint8_t CNF3 = 0x28;
    static constexpr uint8_t TXB0CTRL = 0x30;
    static constexpr uint8_t RXB0CTRL = 0x60;
    static constexpr uint8_t CANINTF = 0x2C;
    static constexpr uint8_t CANINTE = 0x2B;
    static constexpr uint8_t RXB0SIDH = 0x61;
    static constexpr uint8_t RXB0SIDL = 0x62;

};

#endif // MCP2515CONFIGURATOR_HPP
