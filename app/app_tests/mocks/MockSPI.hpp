#ifndef MOCKSPI_HPP
#define MOCKSPI_HPP

#include "SPIController.hpp"
#include <gmock/gmock.h>

class MockSPI : public SPIController
{
public:
    MOCK_METHOD(bool, openDevice, (const std::string &device));
    MOCK_METHOD(void, configure, (uint8_t mode, uint8_t bits, uint32_t speed));
    MOCK_METHOD(void, writeByte, (uint8_t address, uint8_t data));
    MOCK_METHOD(uint8_t, readByte, (uint8_t address));
    MOCK_METHOD(void, spiTransfer, (const uint8_t *tx, uint8_t *rx, size_t length));
    MOCK_METHOD(void, closeDevice, ());
};

#endif // MOCKSPI_HPP
