#include "MockSysCalls.hpp"
#include "SPIController.hpp"
#include <gtest/gtest.h>
#include <linux/spi/spidev.h>

using ::testing::_;
using ::testing::Return;

class SPIControllerTest : public ::testing::Test
{
protected:
    SPIController *spiController;

    void SetUp() override { spiController = new SPIController(mock_ioctl, mock_open, mock_close); }

    void TearDown() override { delete spiController; }
};

TEST_F(SPIControllerTest, OpenDeviceSuccess)
{
    EXPECT_CALL(MockSysCalls::instance(), open(testing::StrEq("/dev/spidev0.0"), O_RDWR))
        .WillOnce(Return(3));

    ASSERT_NO_THROW(spiController->openDevice("/dev/spidev0.0"));
}

TEST_F(SPIControllerTest, OpenDeviceFailure)
{
    EXPECT_CALL(MockSysCalls::instance(), open(testing::StrEq("/dev/spidev0.0"), O_RDWR))
        .WillOnce(Return(-1)); // Simulate failure

    ASSERT_THROW(spiController->openDevice("/dev/spidev0.0"), std::runtime_error);
}

TEST_F(SPIControllerTest, ConfigureSPIValidParameters)
{
    EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
    spiController->openDevice("/dev/spidev0.0");

    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_MODE)).WillOnce(Return(0));
    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_BITS_PER_WORD)).WillOnce(Return(0));
    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_MAX_SPEED_HZ)).WillOnce(Return(0));

    ASSERT_NO_THROW(spiController->configure(0, 8, 500000));
}

TEST_F(SPIControllerTest, WriteByteSuccess)
{
    EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
    spiController->openDevice("/dev/spidev0.0");

    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1))).WillOnce(Return(0));

    ASSERT_NO_THROW(spiController->writeByte(0x01, 0xFF));
}

TEST_F(SPIControllerTest, ReadByteSuccess)
{
    EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
    spiController->openDevice("/dev/spidev0.0");

    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1))).WillOnce(Return(0));

    ASSERT_NO_THROW(spiController->readByte(0x01));
}

TEST_F(SPIControllerTest, SpiTransferSuccess)
{
    EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
    spiController->openDevice("/dev/spidev0.0");

    EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1))).WillOnce(Return(0));

    uint8_t tx[] = {0x02, 0x01, 0xFF};
    uint8_t rx[sizeof(tx)] = {0};

    ASSERT_NO_THROW(spiController->spiTransfer(tx, rx, sizeof(tx)));
}

TEST_F(SPIControllerTest, CloseDeviceSuccess)
{
    EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
    spiController->openDevice("/dev/spidev0.0");

    EXPECT_CALL(MockSysCalls::instance(), close(3)).WillOnce(Return(0));

    ASSERT_NO_THROW(spiController->closeDevice());
}
