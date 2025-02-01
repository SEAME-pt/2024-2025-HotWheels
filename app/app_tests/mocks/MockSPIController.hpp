/**
 * @file MockSPIController.hpp
 * @brief File containing Mock classes to test the SPI controller.
 *
 * This file provides a mock implementation of the SPI controller for testing
 * purposes. It uses Google Mock to create mock methods for SPI operations.
 *
 * @version 0.1
 * @date 2025-01-30
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 * @section License
 * @copyright Copyright (c) 2025
 *
 */

#ifndef MOCKSPICONTROLLER_HPP
#define MOCKSPICONTROLLER_HPP

#include "ISPIController.hpp"
#include <gmock/gmock.h>

/**
 * @class MockSPIController
 * @brief Class to emulate the behavior of the SPI controller. (Overrided the
 * Can0)
 */
class MockSPIController : public ISPIController {
public:
	/** @brief Mocked method to open the SPI device. */
	MOCK_METHOD(bool, openDevice, (const std::string &device), (override));
	/** @brief Mocked method to configure the SPI device. */
	MOCK_METHOD(void, configure, (uint8_t mode, uint8_t bits, uint32_t speed),
							(override));
	/** @brief Mocked method to write a byte of data to the SPI device. */
	MOCK_METHOD(void, writeByte, (uint8_t address, uint8_t data), (override));
	/** @brief Mocked method to read a byte of data from the SPI device. */
	MOCK_METHOD(uint8_t, readByte, (uint8_t address), (override));
	/** @brief Mocked method to transfer data to the SPI device. */
	MOCK_METHOD(void, spiTransfer,
							(const uint8_t *tx, uint8_t *rx, size_t length), (override));
	/** @brief Mocked method to close the SPI device. */
	MOCK_METHOD(void, closeDevice, (), (override));
};

#endif // MOCKSPICONTROLLER_HPP
