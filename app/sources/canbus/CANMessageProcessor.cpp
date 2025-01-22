#include "CANMessageProcessor.hpp"
#include <stdexcept>

CANMessageProcessor::CANMessageProcessor() {}

void CANMessageProcessor::registerHandler(uint16_t frameID, MessageHandler handler) {
    if (!handler) {
        throw std::invalid_argument("Handler cannot be null");
    }
    handlers[frameID] = handler;
}

void CANMessageProcessor::processMessage(uint16_t frameID, const std::vector<uint8_t>& data) {
    auto it = handlers.find(frameID);
    if (it != handlers.end()) {
        it->second(data);
    } else {
        throw std::runtime_error("No handler registered for frame ID: " + std::to_string(frameID));
    }
}
