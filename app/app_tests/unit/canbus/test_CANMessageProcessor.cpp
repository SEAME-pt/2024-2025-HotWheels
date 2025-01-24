#include "CANMessageProcessor.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

class CANMessageProcessorTest : public ::testing::Test {
protected:
    CANMessageProcessor processor;
};

TEST_F(CANMessageProcessorTest, RegisterHandlerSuccess) {
    ASSERT_NO_THROW(processor.registerHandler(0x123, [](const std::vector<uint8_t>&) {}));
}

TEST_F(CANMessageProcessorTest, RegisterHandlerNullThrowsException) {
    ASSERT_THROW(processor.registerHandler(0x123, nullptr), std::invalid_argument);
}

TEST_F(CANMessageProcessorTest, ProcessMessageWithRegisteredHandler) {
    bool handlerCalled = false;
    processor.registerHandler(0x123, [&](const std::vector<uint8_t>& data) {
        handlerCalled = true;
        ASSERT_EQ(data.size(), 2);
        ASSERT_EQ(data[0], 0xA0);
        ASSERT_EQ(data[1], 0xB1);
    });

    std::vector<uint8_t> message = {0xA0, 0xB1};
    processor.processMessage(0x123, message);
    ASSERT_TRUE(handlerCalled);
}

TEST_F(CANMessageProcessorTest, ProcessMessageWithUnregisteredHandlerThrowsException) {
    std::vector<uint8_t> message = {0xA0, 0xB1};
    ASSERT_THROW(processor.processMessage(0x456, message), std::runtime_error);
}

TEST_F(CANMessageProcessorTest, OverwriteHandlerForSameFrameID) {
    bool firstHandlerCalled = false;
    bool secondHandlerCalled = false;

    processor.registerHandler(0x123, [&](const std::vector<uint8_t>&) {
        firstHandlerCalled = true;
    });

    processor.registerHandler(0x123, [&](const std::vector<uint8_t>&) {
        secondHandlerCalled = true;
    });

    std::vector<uint8_t> message = {0xA0, 0xB1};
    processor.processMessage(0x123, message);

    ASSERT_FALSE(firstHandlerCalled);
    ASSERT_TRUE(secondHandlerCalled);
}

