# Integration Tests: MCP2515Controller

This document outlines the integration tests defined in `test_MCP2515Controller.cpp` for validating the `MCP2515Configurator` class. This class manages communication with the MCP2515 CAN controller via SPI using the `SPIController` interface.

## 🧪 Overview

These tests verify:
- Frame transmission and reception over the CAN bus
- Support for different frame types (data, remote, error)
- Communication consistency across various bus speeds

Test framework used:
- [Google Test](https://github.com/google/googletest)

---

## ✅ Test Cases

### 1. `DataFrameTest`
**Purpose:** Verify that a standard CAN data frame can be sent and received correctly.
**Check:**
- Frame ID and data match between sender and receiver
- Assumes CRC and ACK handling is done internally (placeholders used)

---

### 2. `RemoteFrameTest`
**Purpose:** Test handling of remote transmission requests (RTR).
**Check:**
- Frame is correctly received as a remote frame
- Response frame contains expected length (even though mocked)

---

### 3. `ErrorFrameTest`
**Purpose:** Simulate sending a frame with bit errors and ensure retransmission logic is stable.
**Check:**
- Sent and received frames match despite being flagged as error frame
- Re-attempt to send is allowed and succeeds

---

### 4. `MaxBusSpeedTest`
**Purpose:** Check stability of frame transmission at maximum bus speed (1 Mbps).
**Check:**
- Frame ID and content are received correctly
- Internal CRC/ACK assumed to be valid

---

### 5. `MinBusSpeedTest`
**Purpose:** Verify communication reliability at minimum supported speed (e.g., 10 kbps).
**Check:**
- Frame is sent and received successfully
- No corruption occurs at low speed

---

## 🛠 Notes

- These are integration-level tests assuming real hardware or a properly mocked bus layer.
- CRC and ACK validations are not explicitly performed but assumed handled by hardware.
- The tests reuse the same `sendCANMessage` and `readCANMessage` logic, focusing on input/output consistency across frame types and speed configs.
