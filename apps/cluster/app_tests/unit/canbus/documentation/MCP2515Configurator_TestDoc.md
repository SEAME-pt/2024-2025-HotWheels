# Unit Tests: MCP2515Configurator

This document outlines the unit tests defined in `test_MCP2515Configurator.cpp` for the `MCP2515Configurator` class, which configures the MCP2515 CAN controller via SPI.

## 🧪 Overview

These tests verify:
- Correct configuration of chip registers
- Successful and failed chip resets
- Baud rate setup
- TX/RX buffer settings
- Filter/mask configurations
- Interrupts
- Frame read behavior (both with and without data)

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/blob/main/googlemock/README.md)

---

## ✅ Test Cases

### 1. `ResetChipSuccess`
**Purpose:** Ensure `resetChip()` returns true when the controller resets correctly.
**Mocked Behavior:** `spiTransfer` followed by `readByte()` returning expected mode value.

---

### 2. `ResetChipFailure`
**Purpose:** Ensure `resetChip()` returns false when the chip doesn't enter config mode.

---

### 3. `ConfigureBaudRate`
**Purpose:** Confirm that `CNF1`, `CNF2`, and `CNF3` registers are written with expected values for baud rate setup.

---

### 4. `ConfigureTXBuffer`
**Purpose:** Ensure TX buffer control register is correctly written.

---

### 5. `ConfigureRXBuffer`
**Purpose:** Ensure RX buffer control register is correctly configured to accept all valid messages.

---

### 6. `ConfigureFiltersAndMasks`
**Purpose:** Ensure correct filter and mask values are written.
**Note:** Uses hardcoded register addresses (0x00, 0x01).

---

### 7. `ConfigureInterrupts`
**Purpose:** Verify interrupt enable register is set correctly (e.g., `CANINTE` set to `0x01`).

---

### 8. `SetMode`
**Purpose:** Confirm that calling `setMode()` writes to the `CANCTRL` register with the provided mode.

---

### 9. `VerifyModeSuccess`
**Purpose:** Ensure `verifyMode()` returns true when the current mode matches the expected one.

---

### 10. `VerifyModeFailure`
**Purpose:** Ensure `verifyMode()` returns false when mode doesn't match.

---

### 11. `ReadCANMessageWithData`
**Purpose:** Simulate receiving a full CAN message and validate the frame ID and data contents.
**Steps:**
- Simulate `CANINTF` showing message pending
- Return `SIDH`, `SIDL`, DLC, and payload bytes
- Clear interrupt flag and validate the output

---

### 12. `ReadCANMessageNoData`
**Purpose:** Ensure `readCANMessage()` returns an empty vector when no interrupt flag is set.

---

## 🛠 Notes

- All SPI interactions are mocked via `MockSPIController`, allowing isolation of logic.
- Tests use the `CANINTF`, `RXB0SIDH`, `RXB0SIDL`, and data registers directly.
- The `readCANMessage()` function extracts a full CAN frame and uses register math for frame decoding.
