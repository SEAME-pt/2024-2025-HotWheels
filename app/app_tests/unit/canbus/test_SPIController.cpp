/**
 * @file test_SPIController.cpp
 * @brief Unit tests for the SPIController class.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 * @version 0.1
 * @date 2025-01-30
 *
 * @details This file contains unit tests for the SPIController class, using
 * Google Test and Google Mock frameworks.
 */

#include "MockSysCalls.hpp"
#include "SPIController.hpp"
#include <gtest/gtest.h>
#include <linux/spi/spidev.h>

using ::testing::_;
using ::testing::Return;

/**
 * @class SPIControllerTest
 * @brief Test fixture for testing the SPIController class.
 *
 * @details This class sets up the necessary mock objects and provides setup and
 * teardown methods for each test.
 */
class SPIControllerTest : public ::testing::Test {
protected:
  /** @brief SPIController object. */
  SPIController *spiController;

  /**
   * @brief Set up the test environment.
   *
   * @details This method is called before each test to set up the necessary
   * objects.
   */
  void SetUp() override {
    spiController = new SPIController(mock_ioctl, mock_open, mock_close);
  }

  /**
   * @brief Tear down the test environment.
   *
   * @details This method is called after each test to clean up the objects
   * created in SetUp().
   */
  void TearDown() override { delete spiController; }
};

/**
 * @test Tests if the device opens successfully.
 * @brief Ensures that the device opens without throwing an exception.
 *
 * @details This test verifies that openDevice() does not throw an exception
 * when the device opens successfully.
 *
 * @see SPIController::openDevice
 */
TEST_F(SPIControllerTest, OpenDeviceSuccess) {
  EXPECT_CALL(MockSysCalls::instance(),
              open(testing::StrEq("/dev/spidev0.0"), O_RDWR))
      .WillOnce(Return(3));

  ASSERT_NO_THROW(spiController->openDevice("/dev/spidev0.0"));
}

/**
 * @test Tests if the device fails to open.
 * @brief Ensures that openDevice() throws an exception when the device fails to
 * open.
 *
 * @details This test verifies that openDevice() throws a runtime_error when the
 * device fails to open.
 *
 * @see SPIController::openDevice
 */
TEST_F(SPIControllerTest, OpenDeviceFailure) {
  EXPECT_CALL(MockSysCalls::instance(),
              open(testing::StrEq("/dev/spidev0.0"), O_RDWR))
      .WillOnce(Return(-1)); // Simulate failure

  ASSERT_THROW(spiController->openDevice("/dev/spidev0.0"), std::runtime_error);
}

/**
 * @test Tests if the SPI configuration is successful with valid parameters.
 * @brief Ensures that configure() does not throw an exception with valid
 * parameters.
 *
 * @details This test verifies that configure() does not throw an exception when
 * called with valid parameters.
 *
 * @see SPIController::configure
 */
TEST_F(SPIControllerTest, ConfigureSPIValidParameters) {
  EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
  spiController->openDevice("/dev/spidev0.0");

  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_MODE))
      .WillOnce(Return(0));
  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_BITS_PER_WORD))
      .WillOnce(Return(0));
  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_WR_MAX_SPEED_HZ))
      .WillOnce(Return(0));

  ASSERT_NO_THROW(spiController->configure(0, 8, 500000));
}

/**
 * @test Tests if writing a byte is successful.
 * @brief Ensures that writeByte() does not throw an exception.
 *
 * @details This test verifies that writeByte() does not throw an exception when
 * writing a byte.
 *
 * @see SPIController::writeByte
 */
TEST_F(SPIControllerTest, WriteByteSuccess) {
  EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
  spiController->openDevice("/dev/spidev0.0");

  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1)))
      .WillOnce(Return(0));

  ASSERT_NO_THROW(spiController->writeByte(0x01, 0xFF));
}

/**
 * @test Tests if reading a byte is successful.
 * @brief Ensures that readByte() does not throw an exception.
 *
 * @details This test verifies that readByte() does not throw an exception when
 * reading a byte.
 *
 * @see SPIController::readByte
 */
TEST_F(SPIControllerTest, ReadByteSuccess) {
  EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
  spiController->openDevice("/dev/spidev0.0");

  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1)))
      .WillOnce(Return(0));

  ASSERT_NO_THROW(spiController->readByte(0x01));
}

/**
 * @test Tests if SPI transfer is successful.
 * @brief Ensures that spiTransfer() does not throw an exception.
 *
 * @details This test verifies that spiTransfer() does not throw an exception
 * when performing an SPI transfer.
 *
 * @see SPIController::spiTransfer
 */
TEST_F(SPIControllerTest, SpiTransferSuccess) {
  EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
  spiController->openDevice("/dev/spidev0.0");

  EXPECT_CALL(MockSysCalls::instance(), ioctl(_, SPI_IOC_MESSAGE(1)))
      .WillOnce(Return(0));

  uint8_t tx[] = {0x02, 0x01, 0xFF};
  uint8_t rx[sizeof(tx)] = {0};

  ASSERT_NO_THROW(spiController->spiTransfer(tx, rx, sizeof(tx)));
}

/**
 * @test Tests if the device closes successfully.
 * @brief Ensures that closeDevice() does not throw an exception.
 *
 * @details This test verifies that closeDevice() does not throw an exception
 * when the device closes successfully.
 *
 * @see SPIController::closeDevice
 */
TEST_F(SPIControllerTest, CloseDeviceSuccess) {
  EXPECT_CALL(MockSysCalls::instance(), open(_, _)).WillOnce(Return(3));
  spiController->openDevice("/dev/spidev0.0");

  EXPECT_CALL(MockSysCalls::instance(), close(3)).WillOnce(Return(0));

  ASSERT_NO_THROW(spiController->closeDevice());
}
