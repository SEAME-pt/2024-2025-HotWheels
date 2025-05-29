# Unit Tests: MCP2515Controller

This document summarizes the unit tests defined in `test_MCP2515Controller.cpp` for the `MCP2515Controller` class, which handles CAN communication via SPI using the MCP2515 chip.

## Overview

These tests validate:
- Initialization success and failure
- Signal emission for speed and RPM updates
- Handler registration and invocation
- Threaded data reading
- Stop flag behavior

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/blob/main/googlemock/README.md)
- [Qt (QSignalSpy)](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `InitializationSuccess`
**Purpose:** Ensure the controller initializes without throwing.
**Mocks:**
- `openDevice()` returns `true`
- SPI communication and mode checks proceed normally

---

### 2. `InitializationFailure`
**Purpose:** Ensure a failed `openDevice()` throws a `std::runtime_error`.
**Mocks:**
- `openDevice()` returns `false`

---

### 3. `SetupHandlersTest`
**Purpose:** Validate that `registerHandler()` works for different frame IDs.
**Checks:**
- No exceptions are thrown when handlers are registered

---

### 4. `SpeedUpdatedSignal`
**Purpose:** Ensure `speedUpdated(float)` signal is emitted correctly.
**Mocks:**
- Simulates float decoding (e.g., `1.0` from encoded `10.0`)
**Test:**
- Uses `QSignalSpy` to capture the signal and check the value

---

### 5. `RpmUpdatedSignal`
**Purpose:** Ensure `rpmUpdated(int)` signal is emitted with correct value.
**Mocks:**
- Emits signal after processing RPM bytes (`0x03E8` = `1000`)
**Test:**
- Captures signal using `QSignalSpy`

---

### 6. `ProcessReadingCallsHandlers`
**Purpose:** Ensure `processReading()` processes available data.
**Mocks:**
- Simulates available data and SPI response
**Execution:**
- Runs `processReading()` in a thread
- Verifies correct flow and termination

---

### 7. `StopReadingStopsProcessing`
**Purpose:** Ensure `stopReading()` sets the flag as expected.
**Check:**
- `isStopReadingFlagSet()` returns `true`

---

## Notes

- Tests are fully isolated using `MockSPIController`.
- Signal verification is done using Qt's `QSignalSpy`.
- Threading and timing are used in `ProcessReadingCallsHandlers` to mimic real-time behavior.
- Each test clearly distinguishes between setup and behavioral checks.
