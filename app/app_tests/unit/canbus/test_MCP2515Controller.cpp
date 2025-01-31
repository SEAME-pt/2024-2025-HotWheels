#include <QObject>
#include <QSignalSpy>
#include "MCP2515Controller.hpp"
#include "MockSPIController.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

class MCP2515ControllerTest : public ::testing::Test
{
protected:
    MockSPIController mockSPI;
    MCP2515Configurator configurator{mockSPI};
    CANMessageProcessor messageProcessor;

    MCP2515ControllerTest() = default;
};

TEST_F(MCP2515ControllerTest, InitializationSuccess)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);
    EXPECT_CALL(mockSPI, spiTransfer(_, nullptr, 1)).WillOnce(Return());
    EXPECT_CALL(mockSPI, readByte(_)).WillOnce(Return(0x80)).WillRepeatedly(Return(0x00));
    EXPECT_CALL(mockSPI, writeByte(_, _)).Times(::testing::AtLeast(1));

    MCP2515Controller controller("/dev/spidev0.0", mockSPI);
    ASSERT_NO_THROW(controller.init());
}

TEST_F(MCP2515ControllerTest, InitializationFailure)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/nonexistent")).WillOnce(Return(false));
    ASSERT_THROW(MCP2515Controller("/dev/nonexistent", mockSPI), std::runtime_error);
}

TEST_F(MCP2515ControllerTest, SetupHandlersTest)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);
    MCP2515Controller controller("/dev/spidev0.0", mockSPI);
    auto &processor = controller.getMessageProcessor();

    ASSERT_NO_THROW(processor.registerHandler(0x100, [](const std::vector<uint8_t> &) {}));
    ASSERT_NO_THROW(processor.registerHandler(0x200, [](const std::vector<uint8_t> &) {}));
}

TEST_F(MCP2515ControllerTest, SpeedUpdatedSignal)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);
    MCP2515Controller controller("/dev/spidev0.0", mockSPI);

    QSignalSpy speedSpy(&controller, &MCP2515Controller::speedUpdated);
    auto &processor = controller.getMessageProcessor();

    std::vector<uint8_t> data = {0x00, 0x00, 0x20, 0x41}; // Float value: 10.0
    processor.processMessage(0x100, data);

    ASSERT_EQ(speedSpy.count(), 1);
    QList<QVariant> arguments = speedSpy.takeFirst();
    ASSERT_EQ(arguments.at(0).toFloat(), 1.0f); // Speed divided by 10
}

TEST_F(MCP2515ControllerTest, RpmUpdatedSignal)
{
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

TEST_F(MCP2515ControllerTest, ProcessReadingCallsHandlers)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);
    EXPECT_CALL(mockSPI, readByte(_))
        .WillOnce(Return(0x01))        // Indicate data available
        .WillRepeatedly(Return(0x00)); // No more data
    EXPECT_CALL(mockSPI, spiTransfer(_, _, _))
        .WillRepeatedly([](const uint8_t *tx, uint8_t *rx, size_t length) {
            if (length == 3 && tx[0] == 0x03) { // Read command
                rx[1] = 0x12;                   // Frame ID part 1
                rx[2] = 0x34;                   // Frame ID part 2
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

TEST_F(MCP2515ControllerTest, StopReadingStopsProcessing)
{
    EXPECT_CALL(mockSPI, openDevice("/dev/spidev0.0")).WillOnce(Return(true));
    EXPECT_CALL(mockSPI, closeDevice()).Times(1);

    MCP2515Controller controller("/dev/spidev0.0", mockSPI);
    controller.stopReading();
    ASSERT_TRUE(controller.isStopReadingFlagSet());
}
