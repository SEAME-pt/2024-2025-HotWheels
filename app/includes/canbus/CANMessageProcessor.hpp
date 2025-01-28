#ifndef CANMESSAGEPROCESSOR_HPP
#define CANMESSAGEPROCESSOR_HPP

#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>

class CANMessageProcessor {
public:
    using MessageHandler = std::function<void(const std::vector<uint8_t>&)>;

    CANMessageProcessor();
    ~CANMessageProcessor() = default;

    /**
     * Registers a message handler for a specific frame ID.
     *
     * This function associates a given frame ID with a message handler. If the provided
     * handler is null, an exception is thrown to ensure that only valid handlers are registered.
     * Otherwise, the handler is stored in the `handlers` map, allowing the system to process
     * messages associated with the specified frame ID using the registered handler.
     *
     * @param frameID The ID of the message frame for which the handler is being registered.
     * @param handler A function or callable object that will handle messages with the specified frame ID.
     *
     * @throws std::invalid_argument If the handler is null, indicating an invalid registration attempt.
     */
    void registerHandler(uint16_t frameID, MessageHandler handler);

    /**
     * Processes a message by invoking the appropriate handler for the given frame ID.
     *
     * This function searches for a registered handler associated with the provided frame ID.
     * If a handler is found, it is invoked with the provided message data. If no handler is
     * registered for the frame ID, a runtime error is thrown, indicating that the message cannot
     * be processed.
     *
     * @param frameID The ID of the message frame to be processed.
     * @param data A vector containing the message data to be passed to the handler.
     *
     * @throws std::runtime_error If no handler is registered for the given frame ID,
     *         preventing the message from being processed.
     */
    void processMessage(uint16_t frameID, const std::vector<uint8_t>& data);

private:
    std::unordered_map<uint16_t, MessageHandler> handlers;
};

#endif // CANMESSAGEPROCESSOR_HPP
