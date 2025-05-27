# Unit Tests: SPIController

This document provides an overview of the unit tests defined in `test_SPIController.cpp` for the `SPIController` class, which handles low-level SPI communication using custom system call injection for testability.

## Overview

These tests validate:
- Device opening and closing
- SPI configuration
- Byte-level reading and writing
- Full SPI transfers

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/blob/main/googlemock/README.md)

The `MockSysCalls` utility is used to simulate system-level behavior (`open`, `close`, `ioctl`), allowing robust unit testing without requiring actual hardware.

---

## ✅ Test Cases

### 1. `OpenDeviceSuccess`
**Purpose:** Ensure `openDevice()` succeeds when the mock returns a valid file descriptor.
**How:** Mocks `open()` to return `3`.
**Expected:** No exception is thrown.

---

### 2. `OpenDeviceFailure`
**Purpose:** Ensure `openDevice()` throws `std::runtime_error` on failure.
**How:** Mocks `open()` to return `-1`.
**Expected:** Throws an exception.

---

### 3. `ConfigureSPIValidParameters`
**Purpose:** Validate successful SPI configuration.
**Mocks:**
- `ioctl()` for mode, bits per word, and speed set to return 0.
**Expected:** No exception is thrown.

---

### 4. `WriteByteSuccess`
**Purpose:** Ensure `writeByte()` completes successfully.
**Mocks:**
- `ioctl()` for SPI transfer returns 0.
**Expected:** No exception is thrown.

---

### 5. `ReadByteSuccess`
**Purpose:** Ensure `readByte()` completes without error.
**Mocks:**
- `ioctl()` for SPI transfer returns 0.
**Expected:** No exception is thrown.

---

### 6. `SpiTransferSuccess`
**Purpose:** Validate full SPI transfer with given TX buffer.
**Mocks:**
- `ioctl()` simulates transfer success.
**Expected:** No exception is thrown.

---

### 7. `CloseDeviceSuccess`
**Purpose:** Ensure `closeDevice()` closes the file descriptor without error.
**Mocks:**
- `close()` returns 0.
**Expected:** No exception is thrown.

---

## Notes

- The tests rely on `MockSysCalls`, which replaces standard `open`, `close`, and `ioctl` calls to simulate device I/O.
- `SPI_IOC_MESSAGE(n)` is mocked directly to test read/write transactions.
- Tests are isolated and do not interact with real hardware, ensuring portability and safety.
