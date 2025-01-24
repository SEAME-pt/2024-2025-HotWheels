#include "MCP2515Configurator.hpp"
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iostream>

MCP2515Configurator::MCP2515Configurator(ISPIController &spiController)
    : spiController(spiController) {}

bool MCP2515Configurator::resetChip()
{
    sendCommand(RESET_CMD);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    uint8_t status = readRegister(CANSTAT);
    return (status & 0xE0) == 0x80; // Verify configuration mode
}

void MCP2515Configurator::configureBaudRate()
{
    writeRegister(CNF1, 0x00); // Set BRP (Baud Rate Prescaler)
    writeRegister(CNF2, 0x90); // Set Propagation and Phase Segment 1
    writeRegister(CNF3, 0x02); // Set Phase Segment 2
}

void MCP2515Configurator::configureTXBuffer()
{
    writeRegister(TXB0CTRL, 0x00); // Clear TX buffer control register
}

void MCP2515Configurator::configureRXBuffer()
{
    writeRegister(RXB0CTRL, 0x60); // Enable rollover and set RX mode to receive all
}

void MCP2515Configurator::configureFiltersAndMasks()
{
    writeRegister(0x00, 0xFF); // Set filter 0
    writeRegister(0x01, 0xFF); // Set mask 0
}

void MCP2515Configurator::configureInterrupts()
{
    writeRegister(CANINTE, 0x01); // Enable receive interrupt
}

void MCP2515Configurator::setMode(uint8_t mode)
{
    writeRegister(CANCTRL, mode);
}

bool MCP2515Configurator::verifyMode(uint8_t expectedMode)
{
    uint8_t mode = readRegister(CANSTAT) & 0xE0;
    return mode == expectedMode;
}

void MCP2515Configurator::writeRegister(uint8_t address, uint8_t value)
{
    spiController.writeByte(address, value);
}

uint8_t MCP2515Configurator::readRegister(uint8_t address)
{
    return spiController.readByte(address);
}

void MCP2515Configurator::sendCommand(uint8_t command)
{
    uint8_t tx[] = {command};
    spiController.spiTransfer(tx, nullptr, sizeof(tx));
}

std::vector<uint8_t> MCP2515Configurator::readCANMessage(uint16_t &frameID) {
    std::vector<uint8_t> CAN_RX_Buf;
    uint8_t status = readRegister(CAN_RD_STATUS);
    std::cout << "CAN_RD_STATUS: " << std::hex << static_cast<int>(status) << std::endl;

    if (readRegister(CANINTF) & 0x01) { // Check if data is available
        uint8_t sidh = readRegister(RXB0SIDH);
        uint8_t sidl = readRegister(RXB0SIDL);
        frameID = (sidh << 3) | (sidl >> 5);

        uint8_t len = readRegister(RXB0DLC) & 0x0F; // Length of the data
        std::cout << "Received DLC: " << std::hex << static_cast<int>(len) << std::endl;
        for (uint8_t i = 0; i < len; ++i) {
            CAN_RX_Buf.push_back(readRegister(RXB0D0 + i));
            std::cout << "Received Data byte " << std::hex << static_cast<int>(CAN_RX_Buf.back()) << " from RXB0D" << static_cast<int>(i) << std::endl;
        }

        writeRegister(CANINTF, 0x00); // Clear interrupt flag
    }

    return CAN_RX_Buf;
}

void MCP2515Configurator::sendCANMessage(uint16_t frameID, uint8_t* CAN_TX_Buf, uint8_t length1) {
    uint8_t tempdata = readRegister(CAN_RD_STATUS);
    std::cout << "Initial CAN_RD_STATUS: " << std::hex << static_cast<int>(tempdata) << std::endl;

    writeRegister(TXB0SIDH, (frameID >> 3) & 0xFF);
    writeRegister(TXB0SIDL, (frameID & 0x07) << 5);
    std::cout << "Frame ID: " << std::hex << frameID << " written to TXB0SIDH and TXB0SIDL" << std::endl;

    writeRegister(TXB0EID8, 0);
    writeRegister(TXB0EID0, 0);
    writeRegister(TXB0DLC, length1);
    std::cout << "DLC: " << std::hex << static_cast<int>(length1) << " written to TXB0DLC" << std::endl;

    for (uint8_t j = 0; j < length1; ++j) {
        writeRegister(TXB0D0 + j, CAN_TX_Buf[j]);
        std::cout << "Data byte " << std::hex << static_cast<int>(CAN_TX_Buf[j]) << " written to TXB0D" << static_cast<int>(j) << std::endl;
    }

    // Check if TXREQ is set
    if (tempdata & 0x08) { // TXREQ
        std::cout << "TXREQ is set, waiting for it to clear" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // sleep for 0.01 seconds
        writeRegister(TXB0CTRL, 0); // clean flag
        std::cout << "TXREQ flag cleaned" << std::endl;
        while (true) {
            if ((readRegister(CAN_RD_STATUS) & 0x08) == 0) {
                break;
            }
        }
    }

    // Send the RTS command to request transmission
    uint8_t rtsCommand = CAN_RTS_TXB0;
    spiController.spiTransfer(&rtsCommand, nullptr, 1);
    std::cout << "RTS command sent" << std::endl;

    // Verify that the TXREQ bit is set
    uint8_t txCtrlStatus = readRegister(TXB0CTRL);
    std::cout << "TXB0CTRL after RTS command: " << std::hex << static_cast<int>(txCtrlStatus) << std::endl;

    // Verify that the message was sent
    uint8_t txStatus = readRegister(CAN_RD_STATUS);
    std::cout << "Post-transmission CAN_RD_STATUS: " << std::hex << static_cast<int>(txStatus) << std::endl;
}