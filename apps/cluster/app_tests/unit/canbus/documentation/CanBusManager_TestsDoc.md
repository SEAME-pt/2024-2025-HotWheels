# Unit Tests: CanBusManager

This document provides an overview of the unit tests defined in `test_CanBusManager.cpp` for the `CanBusManager` class, using a mocked `MCP2515Controller`.

## Overview

These tests verify:
- Signal forwarding (speed, RPM)
- Initialization logic
- Cleanup behavior in the destructor

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)
- [Qt Signal Testing](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `SpeedSignalEmitsCorrectly`
**Purpose:** Ensure `CanBusManager` forwards speed updates correctly.
**How:** Emits `speedUpdated` from the mock controller and uses `QSignalSpy` to verify emission with correct float value.

---

### 2. `RpmSignalEmitsCorrectly`
**Purpose:** Ensure `rpmUpdated` signal emits the correct integer value.
**How:** Simulates `rpmUpdated` from the mock and validates the output.

---

### 3. `InitializeFailsWhenControllerFails`
**Purpose:** Confirm that `initialize()` returns `false` when the underlying controller's `init()` fails.
**How:** Mocks `init()` to return `false` and asserts the result of `CanBusManager::initialize()`.

---

### 4. `InitializeSucceedsWhenControllerSucceeds`
**Purpose:** Validate that `initialize()` returns `true` when the controller initializes correctly.
**How:** Mocks `init()` to return `true` and asserts a successful initialization.

---

### 5. `DestructorCallsStopReading`
**Purpose:** Ensure the manager calls `stopReading()` on destruction.
**How:** Verifies that the destructor triggers `stopReading()` once after `initialize()` succeeds.

---

## Notes

- The `CanBusManagerTest` fixture owns both the manager and its mock controller.
- All interactions with the controller are handled through the interface `IMCP2515Controller`.
- Cleanup and signal emissions are validated using both `EXPECT_CALL()` (gMock) and `QSignalSpy` (Qt).
