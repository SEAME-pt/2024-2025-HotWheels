#ifndef MOCKSPI_HPP
#define MOCKSPI_HPP

#include "ISPIController.hpp"
#include <gmock/gmock.h>

class MockSPI : public ISPIController {
public:
    MOCK_METHOD(bool, openDevice, (const std::string& device), (override));
    MOCK_METHOD(void, configure, (uint8_t mode, uint8_t bits, uint32_t speed), (override));
    MOCK_METHOD(void, writeByte, (uint8_t address, uint8_t data), (override));
    MOCK_METHOD(uint8_t, readByte, (uint8_t address), (override));
    MOCK_METHOD(void, spiTransfer, (const uint8_t* tx, uint8_t* rx, size_t length), (override));
    MOCK_METHOD(void, closeDevice, (), (override));
};

#endif // MOCKSPI_HPP
