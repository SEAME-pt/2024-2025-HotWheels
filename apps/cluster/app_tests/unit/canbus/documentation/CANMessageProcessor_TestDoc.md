# Unit Tests: CANMessageProcessor

This document outlines the unit tests defined in `test_CANMessageProcessor.cpp` for the `CANMessageProcessor` class, which manages dynamic registration and invocation of message handlers for specific CAN frame IDs.

## Overview

These tests validate:
- Handler registration
- Handler invocation with valid and invalid inputs
- Overwriting of handlers
- Exception handling for null or missing handlers

Test framework used:
- [Google Test](https://github.com/google/googletest)

---

## ✅ Test Cases

### 1. `RegisterHandlerSuccess`
**Purpose:** Ensure that a valid handler can be registered without throwing.
**How:** Calls `registerHandler()` with a lambda and checks for no exceptions.

---

### 2. `RegisterHandlerNullThrowsException`
**Purpose:** Verify that registering a `nullptr` handler triggers an `invalid_argument`.
**How:** Calls `registerHandler()` with `nullptr` and expects an exception.

---

### 3. `ProcessMessageWithRegisteredHandler`
**Purpose:** Ensure the registered handler is invoked with the expected data.
**How:** Registers a lambda, processes a message, and checks that it was called with the correct byte values.

---

### 4. `ProcessMessageWithUnregisteredHandlerThrowsException`
**Purpose:** Ensure `processMessage()` throws if no handler is registered.
**How:** Attempts to process a message for a non-existent frame ID and expects a `runtime_error`.

---

### 5. `OverwriteHandlerForSameFrameID`
**Purpose:** Ensure registering a new handler for the same frame ID replaces the previous one.
**How:** Registers two different handlers for the same ID and verifies that only the second one is called.

---

## Notes

- The tests use `std::vector<uint8_t>` to simulate CAN message payloads.
- Frame ID `0x123` is commonly used as a testable example across tests.
- No external dependencies or mocks are used — tests validate the internal behavior of the handler map.
