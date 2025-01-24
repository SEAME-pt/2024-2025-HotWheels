#include "MCP2515Configurator.hpp"
#include "MockSPIController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

class MCP2515ConfiguratorTest : public ::testing::Test
{
protected:
    MockSPIController mockSPI;
    MCP2515Configurator configurator{mockSPI};
};

TEST_F(MCP2515ConfiguratorTest, ResetChipSuccess)
{
    EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).Times(1);
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT)).WillOnce(Return(0x80));
    ASSERT_TRUE(configurator.resetChip());
}

TEST_F(MCP2515ConfiguratorTest, ResetChipFailure)
{
    EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).Times(1);
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT)).WillOnce(Return(0x00));
    ASSERT_FALSE(configurator.resetChip());
}

TEST_F(MCP2515ConfiguratorTest, ConfigureBaudRate)
{
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF1, 0x00)).Times(1);
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF2, 0x90)).Times(1);
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CNF3, 0x02)).Times(1);
    configurator.configureBaudRate();
}

TEST_F(MCP2515ConfiguratorTest, ConfigureTXBuffer)
{
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::TXB0CTRL, 0x00)).Times(1);
    configurator.configureTXBuffer();
}

TEST_F(MCP2515ConfiguratorTest, ConfigureRXBuffer)
{
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::RXB0CTRL, 0x60)).Times(1);
    configurator.configureRXBuffer();
}

TEST_F(MCP2515ConfiguratorTest, ConfigureFiltersAndMasks)
{
    EXPECT_CALL(mockSPI, writeByte(0x00, 0xFF)).Times(1);
    EXPECT_CALL(mockSPI, writeByte(0x01, 0xFF)).Times(1);
    configurator.configureFiltersAndMasks();
}

TEST_F(MCP2515ConfiguratorTest, ConfigureInterrupts)
{
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CANINTE, 0x01)).Times(1);
    configurator.configureInterrupts();
}

TEST_F(MCP2515ConfiguratorTest, SetMode)
{
    EXPECT_CALL(mockSPI, writeByte(MCP2515Configurator::CANCTRL, 0x02)).Times(1);
    configurator.setMode(0x02);
}

TEST_F(MCP2515ConfiguratorTest, VerifyModeSuccess)
{
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT)).WillOnce(Return(0x80));
    ASSERT_TRUE(configurator.verifyMode(0x80));
}

TEST_F(MCP2515ConfiguratorTest, VerifyModeFailure)
{
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANSTAT)).WillOnce(Return(0x00));
    ASSERT_FALSE(configurator.verifyMode(0x80));
}

TEST_F(MCP2515ConfiguratorTest, ReadCANMessageWithData)
{
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANINTF)).WillOnce(Return(0x01));
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::RXB0SIDH)).WillOnce(Return(0x10));
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::RXB0SIDL)).WillOnce(Return(0x20));
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

TEST_F(MCP2515ConfiguratorTest, ReadCANMessageNoData)
{
    EXPECT_CALL(mockSPI, readByte(MCP2515Configurator::CANINTF)).WillOnce(Return(0x00));
    uint16_t frameID;
    auto data = configurator.readCANMessage(frameID);
    ASSERT_TRUE(data.empty());
}
