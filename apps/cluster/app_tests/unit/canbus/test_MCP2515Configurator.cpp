/*!
 * @file test_MCP2515Configurator.cpp
 * @brief Unit tests for the MCP2515Configurator class.
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains unit tests for the MCP2515Configurator class,
 * using Google Test and Google Mock frameworks.
 */

#include "MCP2515Configurator.hpp"
#include "MockSPIController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

/*!
 * @class MCP2515ConfiguratorTest
 * @brief Test fixture for testing the MCP2515Configurator class.
 *
 * @details This class sets up the necessary mock objects and provides setup and
 * teardown methods for each test.
 */
class MCP2515ConfiguratorTest : public ::testing::Test {
protected:
	/*! @brief Mocked SPI controller. */
	MockSPIController mockSPI;
	/*! @brief MCP2515Configurator object. */
	MCP2515Configurator configurator{mockSPI};
};

/*!
 * @test Tests if the chip reset is successful.
 * @brief Ensures that resetChip() returns true when the reset is successful.
 * @details Verifies that resetChip() returns true when the chip reset is
 * successful.
 * @see MCP2515Configurator::resetChip
 */
TEST_F(MCP2515ConfiguratorTest, ResetChipSuccess) {
	EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).Times(1);
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT))
			.WillOnce(Return(0x80));
	ASSERT_TRUE(configurator.resetChip());
}

/*!
 * @test Tests if the chip reset fails.
 * @brief Ensures that resetChip() returns false when the reset fails.
 * @details Verifies that resetChip() returns false when the chip reset fails.
 * @see MCP2515Configurator::resetChip
 */
TEST_F(MCP2515ConfiguratorTest, ResetChipFailure) {
	EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).Times(1);
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT))
			.WillOnce(Return(0x00));
	ASSERT_FALSE(configurator.resetChip());
}

/*!
 * @test Tests if the baud rate is configured correctly.
 * @brief Ensures that configureBaudRate() writes the correct values to the
 * registers.
 * @details Verifies that configureBaudRate() writes the correct values to the
 * CNF1, CNF2, and CNF3 registers.
 * @see MCP2515Configurator::configureBaudRate
 */
TEST_F(MCP2515ConfiguratorTest, ConfigureBaudRate) {
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF1, 0x00)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF2, 0x90)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF3, 0x02)).Times(1);
	configurator.configureBaudRate();
}

/*!
 * @test Tests if the TX buffer is configured correctly.
 * @brief Ensures that configureTXBuffer() writes the correct value to the
 * TXB0CTRL register.
 * @details Verifies that configureTXBuffer() writes the correct value to the
 * TXB0CTRL register.
 * @see MCP2515Configurator::configureTXBuffer
 */
TEST_F(MCP2515ConfiguratorTest, ConfigureTXBuffer) {
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0CTRL, 0x00)).Times(1);
	configurator.configureTXBuffer();
}

/*!
 * @test Tests if the RX buffer is configured correctly.
 * @brief Ensures that configureRXBuffer() writes the correct value to the
 * RXB0CTRL register.
 * @details Verifies that configureRXBuffer() writes the correct value to the
 * RXB0CTRL register.
 * @see MCP2515Configurator::configureRXBuffer
 */
TEST_F(MCP2515ConfiguratorTest, ConfigureRXBuffer) {
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::RXB0CTRL, 0x60)).Times(1);
	configurator.configureRXBuffer();
}

/*!
 * @test Tests if the filters and masks are configured correctly.
 * @brief Ensures that configureFiltersAndMasks() writes the correct values to
 * the registers.
 * @details Verifies that configureFiltersAndMasks() writes the correct values
 * to the registers.
 * @see MCP2515Configurator::configureFiltersAndMasks
 */
TEST_F(MCP2515ConfiguratorTest, ConfigureFiltersAndMasks) {
	EXPECT_CALL(mockSPI, writeByte(0x00, 0xFF)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(0x01, 0xFF)).Times(1);
	configurator.configureFiltersAndMasks();
}

/*!
 * @test Tests if the interrupts are configured correctly.
 * @brief Ensures that configureInterrupts() writes the correct value to the
 * CANINTE register.
 * @details Verifies that configureInterrupts() writes the correct value to the
 * CANINTE register.
 * @see MCP2515Configurator::configureInterrupts
 */
TEST_F(MCP2515ConfiguratorTest, ConfigureInterrupts) {
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CANINTE, 0x01)).Times(1);
	configurator.configureInterrupts();
}

/*!
 * @test Tests if the mode is set correctly.
 * @brief Ensures that setMode() writes the correct value to the CANCTRL
 * register.
 * @details Verifies that setMode() writes the correct value to the CANCTRL
 * register.
 * @see MCP2515Configurator::setMode
 */
TEST_F(MCP2515ConfiguratorTest, SetMode) {
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CANCTRL, 0x02)).Times(1);
	configurator.setMode(0x02);
}

/*!
 * @test Tests if the mode verification is successful.
 * @brief Ensures that verifyMode() returns true when the mode is correct.
 * @details Verifies that verifyMode() returns true when the mode is correct.
 * @see MCP2515Configurator::verifyMode
 */
TEST_F(MCP2515ConfiguratorTest, VerifyModeSuccess) {
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT))
			.WillOnce(Return(0x80));
	ASSERT_TRUE(configurator.verifyMode(0x80));
}

/*!
 * @test Tests if the mode verification fails.
 * @brief Ensures that verifyMode() returns false when the mode is incorrect.
 * @details Verifies that verifyMode() returns false when the mode is incorrect.
 * @see MCP2515Configurator::verifyMode
 */
TEST_F(MCP2515ConfiguratorTest, VerifyModeFailure) {
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT))
			.WillOnce(Return(0x00));
	ASSERT_FALSE(configurator.verifyMode(0x80));
}

/*!
 * @test Tests if a CAN message with data is read correctly.
 * @brief Ensures that readCANMessage() reads the correct frame ID and data.
 * @details Verifies that readCANMessage() reads the correct frame ID and data
 * when a CAN message with data is available.
 * @see MCP2515Configurator::readCANMessage
 */
TEST_F(MCP2515ConfiguratorTest, ReadCANMessageWithData) {
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANINTF))
			.WillOnce(Return(0x01));
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::RXB0SIDH))
			.WillOnce(Return(0x10));
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::RXB0SIDL))
			.WillOnce(Return(0x20));
	EXPECT_CALL(mockSPI, readByte(0x65)).WillOnce(Return(3));
	EXPECT_CALL(mockSPI, readByte(0x66)).WillOnce(Return(0xA0));
	EXPECT_CALL(mockSPI, readByte(0x67)).WillOnce(Return(0xB1));
	EXPECT_CALL(mockSPI, readByte(0x68)).WillOnce(Return(0xC2));
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CANINTF, 0x00)).Times(1);

	uint16_t frameID;
	auto data = configurator.readCANMessage(frameID);
	ASSERT_EQ(frameID, 0x81);
	ASSERT_EQ(data.size(), 3);
	ASSERT_EQ(data[0], 0xA0);
	ASSERT_EQ(data[1], 0xB1);
	ASSERT_EQ(data[2], 0xC2);
}

/*!
 * @test Tests if a CAN message with no data is read correctly.
 * @brief Ensures that readCANMessage() returns an empty data vector when no CAN
 * message is available.
 * @details Verifies that readCANMessage() returns an empty data vector when no
 * CAN message is available.
 * @see MCP2515Configurator::readCANMessage
 */
TEST_F(MCP2515ConfiguratorTest, ReadCANMessageNoData) {
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANINTF))
			.WillOnce(Return(0x00));
	uint16_t frameID;
	auto data = configurator.readCANMessage(frameID);
	ASSERT_TRUE(data.empty());
}

/*!
 * @test Tests sending a CAN message using sendCANMessage().
 * @brief Ensures correct register writes and RTS command execution.
 *
 * @see MCP2515Configurator::sendCANMessage
 */
TEST_F(MCP2515ConfiguratorTest, SendCANMessage_Success) {
	const uint16_t frameID = 0x123;
	uint8_t messageData[3] = {0xA1, 0xB2, 0xC3};
	uint8_t length = 3;

	// Simulate TXREQ bit set in CAN_RD_STATUS (0x04)
	EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CAN_RD_STATUS))
		.WillOnce(Return(0x04)) // Before loop
		.WillRepeatedly(Return(0x00)); // Clears in loop

	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0SIDH, (frameID >> 3) & 0xFF)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0SIDL, (frameID & 0x07) << 5)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0EID8, 0)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0EID0, 0)).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0DLC, length)).Times(1);

	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0D0, messageData[0])).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0D0 + 1, messageData[1])).Times(1);
	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0D0 + 2, messageData[2])).Times(1);

	EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0CTRL, 0x00)).Times(1);

	EXPECT_CALL(mockSPI, spiTransfer(::testing::NotNull(), nullptr, 1)).Times(1);

	configurator.sendCANMessage(frameID, messageData, length);
}
