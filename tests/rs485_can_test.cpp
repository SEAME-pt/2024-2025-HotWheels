#include <gtest/gtest.h>
#include "MCP2515.hpp"

class RS485CANTest : public ::testing::Test {
protected:
    MCP2515* canBusController;

    void SetUp() override {
        canBusController = new MCP2515("/dev/spidev0.0");
        ASSERT_TRUE(canBusController->init());
    }

    void TearDown() override {
        delete canBusController;
    }
};

TEST_F(RS485CANTest, SendCANMessage) {
    std::vector<uint8_t> message = {0x01, 0x02, 0x03, 0x04};
    canBusController->send(message);
    // Add verification code to check if the message was sent correctly
}

TEST_F(RS485CANTest, ReceiveCANMessage) {
    std::vector<uint8_t> receivedMessage = canBusController->receive();
    ASSERT_FALSE(receivedMessage.empty());
    // Add verification code to check if the received message is correct
}

TEST_F(RS485CANTest, SpeedUpdate) {
    std::vector<uint8_t> speedMessage = {0x50}; // Example speed message
    canBusController->send(speedMessage);
    // Add verification code to check if the speed update was received correctly
}

TEST_F(RS485CANTest, RPMUpdate) {
    std::vector<uint8_t> rpmMessage = {0x12, 0x34}; // Example RPM message
    canBusController->send(rpmMessage);
    // Add verification code to check if the RPM update was received correctly
}

TEST_F(RS485CANTest, SignalCadence) {
    std::vector<uint8_t> message = {0x01, 0x02, 0x03, 0x04};
    canBusController->send(message);
    // Add verification code to check the cadence of the signal sent and received
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
