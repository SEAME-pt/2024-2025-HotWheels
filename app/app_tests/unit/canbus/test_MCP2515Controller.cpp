/*!
 * @file test_MCP2515Controller.cpp
 * @brief Unit tests for the MCP2515Controller class.
 * @version 0.1
 * @date 2025-01-30
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 *
 * @details This file contains unit tests for the MCP2515Controller class, using
 * Google Test and Google Mock frameworks.
 */

#include "MCP2515Controller.hpp"
#include "MockSPIController.hpp"
#include <QObject>
#include <QSignalSpy>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

/*!
 * @class MCP2515ControllerTest
 * @brief Test fixture for testing the MCP2515Controller class.
 *
 * @details This class sets up the necessary mock objects and provides setup and
 * teardown methods for each test.
 */
class MCP2515ControllerTest : public ::testing::Test {
protected:
  /*! @brief Mocked SPI controller. */
  MockSPIController mockSPI;
  /*! @brief MCP2515Configurator object. */
  MCP2515Configurator configurator{mockSPI};
  /*! @brief CANMessageProcessor object. */
  CANMessageProcessor messageProcessor;
  /*! @brief MCP2515Controller object set as default. */
  MCP2515ControllerTest() = default;
};

/*!
 * @test Tests if the initialization is successful.
 * @brief Ensures that init() does not throw an exception.
 * @details Verifies that init() does not throw an exception when the
 * initialization is successful.
 * @see MCP2515Controller::init
 */
TEST_F(MCP2515ControllerTest, InitializationSuccess) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);
  EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).WillOnce(Return());
  EXPECT_CALL(mockSPI, readByte(_))
      .WillOnce(Return(0x80))        // For resetChip
      .WillRepeatedly(Return(0x00)); // For verifyMode
  EXPECT_CALL(mockSPI, writeByte(_, _)).Times(::testing::AtLeast(1));

  MCP2515Controller controller("/dev/spidev0.0", mockSPI);
  ASSERT_NO_THROW(controller.init());
}

/*!
 * @test Tests if the initialization fails.
 * @brief Ensures that init() throws an exception when the initialization fails.
 * @details Verifies that init() throws a runtime_error when the initialization
 * fails.
 * @see MCP2515Controller::init
 */
TEST_F(MCP2515ControllerTest, InitializationFailure) {
  EXPECT_CALL(mockSPI, openDevice("/dev/nonexistent")).WillOnce(Return(false));
  ASSERT_THROW(MCP2515Controller("/dev/nonexistent", mockSPI),
               std::runtime_error);
}

/*!
 * @test Tests if handlers are set up correctly.
 * @brief Ensures that registerHandler() does not throw an exception.
 * @details Verifies that registerHandler() does not throw an exception when
 * setting up handlers.
 * @see CANMessageProcessor::registerHandler
 */
TEST_F(MCP2515ControllerTest, SetupHandlersTest) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);
  MCP2515Controller controller("/dev/spidev0.0", mockSPI);
  auto &processor = controller.getMessageProcessor();

  ASSERT_NO_THROW(
      processor.registerHandler(0x100, [](const std::vector<uint8_t> &) {}));
  ASSERT_NO_THROW(
      processor.registerHandler(0x200, [](const std::vector<uint8_t> &) {}));
}

/*!
 * @test Tests if the speedUpdated signal is emitted correctly.
 * @brief Ensures that the speed signal is emitted with the correct value.
 * @details Uses QSignalSpy to verify that speedUpdated emits the expected speed
 * value.
 * @see MCP2515Controller::speedUpdated
 */
TEST_F(MCP2515ControllerTest, SpeedUpdatedSignal) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);
  MCP2515Controller controller("/dev/spidev0.0", mockSPI);

  QSignalSpy speedSpy(&controller, &MCP2515Controller::speedUpdated);
  auto &processor = controller.getMessageProcessor();

  std::vector<uint8_t> data = {0x00, 0x00, 0x20, 0x41}; // Float value: 10.0
  processor.processMessage(0x100, data);

  ASSERT_EQ(speedSpy.count(), 1);
  QList<QVariant> arguments = speedSpy.takeFirst();
  ASSERT_EQ(arguments.at(0).toFloat(), 1.0F); // Speed divided by 10
}

/*!
 * @test Tests if the rpmUpdated signal is emitted correctly.
 * @brief Ensures that the RPM signal emits the correct value.
 * @details Uses QSignalSpy to verify that rpmUpdated emits the expected RPM
 * value.
 * @see MCP2515Controller::rpmUpdated
 */
TEST_F(MCP2515ControllerTest, RpmUpdatedSignal) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);
  MCP2515Controller controller("/dev/spidev0.0", mockSPI);

  QSignalSpy rpmSpy(&controller, &MCP2515Controller::rpmUpdated);
  auto &processor = controller.getMessageProcessor();

  std::vector<uint8_t> data = {0x03, 0xE8}; // Integer value: 1000 RPM
  processor.processMessage(0x200, data);

  ASSERT_EQ(rpmSpy.count(), 1);
  QList<QVariant> arguments = rpmSpy.takeFirst();
  ASSERT_EQ(arguments.at(0).toInt(), 1000);
}

/*!
 * @test Tests if processReading() calls handlers correctly.
 * @brief Ensures that processReading() calls the registered handlers.
 * @details Verifies that processReading() calls the registered handlers when
 * data is available.
 * @see MCP2515Controller::processReading
 */
TEST_F(MCP2515ControllerTest, ProcessReadingCallsHandlers) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);
  EXPECT_CALL(mockSPI, readByte(_))
      .WillOnce(Return(0x01))        // Indicate data available
      .WillRepeatedly(Return(0x00)); // No more data
  EXPECT_CALL(mockSPI, spiTransfer(_, _, _))
      .WillRepeatedly([](const uint8_t *tx, uint8_t *rx, size_t length) {
        if (length == 3 && tx[0] == 0x03) { // Read command
          rx[1] = 0x12;                     // Frame ID part 1
          rx[2] = 0x34;                     // Frame ID part 2
        }
      });
  EXPECT_CALL(mockSPI, writeByte(_, _)).Times(::testing::AtLeast(1));

  MCP2515Controller controller("/dev/spidev0.0", mockSPI);

  std::thread readerThread([&controller]() { controller.processReading(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  controller.stopReading();

  readerThread.join();

  ASSERT_TRUE(controller.isStopReadingFlagSet());
}

/*!
 * @test Tests if stopReading() stops the processing.
 * @brief Ensures that stopReading() sets the stop flag.
 * @details Verifies that stopReading() sets the stop flag to true.
 * @see MCP2515Controller::stopReading
 */
TEST_F(MCP2515ControllerTest, StopReadingStopsProcessing) {
  EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
  EXPECT_CALL(mockSPI, closeDevice()).Times(1);

  MCP2515Controller controller("/dev/spidev0.0", mockSPI);
  controller.stopReading();
  ASSERT_TRUE(controller.isStopReadingFlagSet());
}
