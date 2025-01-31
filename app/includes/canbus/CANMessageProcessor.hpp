/**
 * @file CANMessageProcessor.hpp
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @brief
 * @version 0.1
 * @date 2025-01-31
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef CANMESSAGEPROCESSOR_HPP
#define CANMESSAGEPROCESSOR_HPP

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

class CANMessageProcessor {
public:
  using MessageHandler = std::function<void(const std::vector<uint8_t> &)>;

  CANMessageProcessor();
  ~CANMessageProcessor() = default;
  void registerHandler(uint16_t frameID, MessageHandler &handler);
  void processMessage(uint16_t frameID, const std::vector<uint8_t> &data);

private:
  std::unordered_map<uint16_t, MessageHandler> handlers;
};

#endif // CANMESSAGEPROCESSOR_HPP
