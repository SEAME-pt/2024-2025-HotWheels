#include "MCP2515Configurator.hpp"
#include <thread>
#include <chrono>

MCP2515Configurator::MCP2515Configurator(ISPIController &spiController)
    : spiController(spiController)
{}

bool MCP2515Configurator::resetChip() {
    sendCommand(RESET_CMD);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    uint8_t status = readRegister(CANSTAT);
    return (status & 0xE0) == 0x80;  // Verify configuration mode
}

void MCP2515Configurator::configureBaudRate() {
    writeRegister(CNF1, 0x00);  // Set BRP (Baud Rate Prescaler)
    writeRegister(CNF2, 0x90);  // Set Propagation and Phase Segment 1
    writeRegister(CNF3, 0x02);  // Set Phase Segment 2
}

void MCP2515Configurator::configureTXBuffer() {
    writeRegister(TXB0CTRL, 0x00);  // Clear TX buffer control register
}

void MCP2515Configurator::configureRXBuffer() {
    writeRegister(RXB0CTRL, 0x60);  // Enable rollover and set RX mode to receive all
}

void MCP2515Configurator::configureFiltersAndMasks() {
    writeRegister(0x00, 0xFF);  // Set filter 0
    writeRegister(0x01, 0xFF);  // Set mask 0
}

void MCP2515Configurator::configureInterrupts() {
    writeRegister(CANINTE, 0x01);  // Enable receive interrupt
}

void MCP2515Configurator::setMode(uint8_t mode) {
    writeRegister(CANCTRL, mode);
}

bool MCP2515Configurator::verifyMode(uint8_t expectedMode) {
    uint8_t mode = readRegister(CANSTAT) & 0xE0;
    return mode == expectedMode;
}

void MCP2515Configurator::writeRegister(uint8_t address, uint8_t value) {
    spiController.writeByte(address, value);
}

uint8_t MCP2515Configurator::readRegister(uint8_t address) {
    return spiController.readByte(address);
}

void MCP2515Configurator::sendCommand(uint8_t command) {
    uint8_t tx[] = {command};
    spiController.spiTransfer(tx, nullptr, sizeof(tx));
}

std::vector<uint8_t> MCP2515Configurator::readCANMessage(uint16_t& frameID) {
    std::vector<uint8_t> CAN_RX_Buf;

    if (readRegister(CANINTF) & 0x01) {  // Check if data is available
        uint8_t sidh = readRegister(RXB0SIDH);
        uint8_t sidl = readRegister(RXB0SIDL);
        frameID = (sidh << 3) | (sidl >> 5);

        uint8_t len = readRegister(0x65);  // Length of the data
        for (uint8_t i = 0; i < len; ++i) {
            CAN_RX_Buf.push_back(readRegister(0x66 + i));
        }

        writeRegister(CANINTF, 0x00);  // Clear interrupt flag
    }

    return CAN_RX_Buf;
}

