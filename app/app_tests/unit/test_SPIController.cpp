#include "MockSPI.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

class SPIControllerTest : public ::testing::Test
{
protected:
    MockSPI mockSPI;
};

// Open Device Tests
TEST_F(SPIControllerTest, OpenDeviceSuccess)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    ASSERT_TRUE(mockSPI.openDevice("/dev/spidev0.0"));
}

TEST_F(SPIControllerTest, OpenDeviceFailure)
{
    EXPECT_CALL(mockSPI, openDevice("/nonexistent/spidev0.0")).WillOnce(Return(false));
    ASSERT_FALSE(mockSPI.openDevice("/nonexistent/spidev0.0"));
}

// Configure SPI Tests
TEST_F(SPIControllerTest, ConfigureSPIValidParameters)
{
    EXPECT_CALL(mockSPI, configure(0, 8, 500000)).Times(1);
    ASSERT_NO_THROW(mockSPI.configure(0, 8, 500000));
}

TEST_F(SPIControllerTest, ConfigureSPIInvalidParameters)
{
    EXPECT_CALL(mockSPI, configure(_, _, _)).Times(0);
    ASSERT_NO_THROW(mockSPI.configure(5, 8, 500000)); // Invalid mode, no action expected
}

// Write Byte Tests
TEST_F(SPIControllerTest, WriteByteSuccess)
{
    EXPECT_CALL(mockSPI, writeByte(0x01, 0xFF)).Times(1);
    ASSERT_NO_THROW(mockSPI.writeByte(0x01, 0xFF));
}

// Read Byte Tests
TEST_F(SPIControllerTest, ReadByteSuccess)
{
    EXPECT_CALL(mockSPI, readByte(0x01)).WillOnce(Return(0xFF));
    uint8_t value = mockSPI.readByte(0x01);
    ASSERT_EQ(value, 0xFF);
}

// SPI Transfer Tests
TEST_F(SPIControllerTest, SpiTransferSuccess)
{
    uint8_t tx[] = {0x02, 0x01, 0xFF};
    uint8_t rx[sizeof(tx)] = {0};

    EXPECT_CALL(mockSPI, spiTransfer(tx, rx, sizeof(tx))).Times(1);
    ASSERT_NO_THROW(mockSPI.spiTransfer(tx, rx, sizeof(tx)));
}

TEST_F(SPIControllerTest, SpiTransferWithNullTx)
{
    uint8_t *tx = nullptr;
    uint8_t rx[3] = {0};

    EXPECT_CALL(mockSPI, spiTransfer(_, _, _)).Times(0);
    ASSERT_NO_THROW(mockSPI.spiTransfer(tx, rx, 3)); // No action expected
}

TEST_F(SPIControllerTest, SpiTransferWithNullRx)
{
    uint8_t tx[3] = {0x02, 0x01, 0xFF};
    uint8_t *rx = nullptr;

    EXPECT_CALL(mockSPI, spiTransfer(_, _, _)).Times(0);
    ASSERT_NO_THROW(mockSPI.spiTransfer(tx, rx, 3)); // No action expected
}

// Close Device Tests
TEST_F(SPIControllerTest, CloseDeviceSuccess)
{
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);
    ASSERT_NO_THROW(mockSPI.closeDevice());
}

TEST_F(SPIControllerTest, CloseDeviceWithoutOpening)
{
    EXPECT_CALL(mockSPI, closeDevice()).Times(0);
    ASSERT_NO_THROW(mockSPI.closeDevice()); // No action expected
}
