/**
 * @file test_CANMessageProcessor.cpp
 * @brief Unit tests for the CANMessageProcessor class.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 * @author Michel Batista (@MicchelFAB)
 * @version 0.1
 * @date 2025-01-30
 * 
 * @details This file contains unit tests for the CANMessageProcessor class, using Google Test framework.
 */

#include "CANMessageProcessor.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

/**
 * @class CANMessageProcessorTest
 * @brief Test fixture for testing the CANMessageProcessor class.
 * 
 * @details This class sets up the necessary objects and provides setup and teardown methods for each test.
 */
class CANMessageProcessorTest : public ::testing::Test {
protected:
    CANMessageProcessor processor; ///< CANMessageProcessor object.
};

/**
 * @test Tests if a handler can be registered successfully.
 * @brief Ensures that registerHandler() does not throw an exception.
 * 
 * @details This test verifies that registerHandler() does not throw an exception when a valid handler is registered.
 * 
 * @see CANMessageProcessor::registerHandler
 */
TEST_F(CANMessageProcessorTest, RegisterHandlerSuccess) {
    ASSERT_NO_THROW(processor.registerHandler(0x123, [](const std::vector<uint8_t>&) {}));
}

/**
 * @test Tests if registering a null handler throws an exception.
 * @brief Ensures that registerHandler() throws an invalid_argument exception.
 * 
 * @details This test verifies that registerHandler() throws an invalid_argument exception when a null handler is registered.
 * 
 * @see CANMessageProcessor::registerHandler
 */
TEST_F(CANMessageProcessorTest, RegisterHandlerNullThrowsException) {
    ASSERT_THROW(processor.registerHandler(0x123, nullptr), std::invalid_argument);
}

/**
 * @test Tests if a message is processed with a registered handler.
 * @brief Ensures that the registered handler is called with the correct data.
 * 
 * @details This test verifies that processMessage() calls the registered handler with the correct data.
 * 
 * @see CANMessageProcessor::processMessage
 */
TEST_F(CANMessageProcessorTest, ProcessMessageWithRegisteredHandler) {
    bool handlerCalled = false;
    processor.registerHandler(0x123, [&](const std::vector<uint8_t>& data) {
        handlerCalled = true;
        ASSERT_EQ(data.size(), 2);
        ASSERT_EQ(data[0], 0xA0);
        ASSERT_EQ(data[1], 0xB1);
    });

 
    processor.processMessage(0x123, message);
    ASSERT_TRUE(handlerCalled);
}

/**
 * @test Tests if processing a message with an unregistered handler throws an exception.
 * @brief Ensures that processMessage() throws a runtime_error exception.
 * 
 * @details This test verifies that processMessage() throws a runtime_error exception when no handler is registered for the given frame ID.
 * 
 * @see CANMessageProcessor::processMessage
 */
TEST_F(CANMessageProcessorTest, ProcessMessageWithUnregisteredHandlerThrowsException) {
    std::vector<uint8_t> message = {0xA0, 0xB1};
    ASSERT_THROW(processor.processMessage(0x456, message), std::runtime_error);
}

/**
 * @test Tests if a handler can be overwritten for the same frame ID.
 * @brief Ensures that the new handler is called instead of the old handler.
 * 
 * @details This test verifies that registering a new handler for the same frame ID overwrites the old handler.
 * 
 * @see CANMessageProcessor::registerHandler
 */
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

