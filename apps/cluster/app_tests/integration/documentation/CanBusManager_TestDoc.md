# Integration Tests: CanBusManager

This document provides an overview of the integration tests defined in `test_int_CanBusManager.cpp` for the `CanBusManager` class, which interfaces with the MCP2515 CAN controller.

## Overview

These tests verify:
- Signal forwarding from the controller to the manager
- Correct initialization of the CAN manager
- Proper thread handling and cleanup

Test framework used:
- [Google Test](https://github.com/google/googletest)
- [Qt Test Utilities](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `ForwardSpeedDataFromMCP2515`
**Purpose:** Ensure the `CanBusManager` correctly forwards speed data from the MCP2515 controller.
**How:** Emits a fake `speedUpdated` signal from the controller and verifies the manager re-emits it using `QSignalSpy`.

### 2. `ForwardRpmDataFromMCP2515`
**Purpose:** Ensure the manager forwards RPM data emitted by the controller.
**How:** Similar to speed, a fake `rpmUpdated` signal is emitted and its propagation is tested.

### 3. `InitializeCanBusManager`
**Purpose:** Test the `initialize()` method of `CanBusManager`.
**Checks:**
- The method returns `true`
- A thread is correctly created
- The thread is running

### 4. `ManagerCleanUpBehavior`
**Purpose:** Test if resources are cleaned up properly upon object deletion.
**Steps:**
- Create a `CanBusManager`
- Initialize and verify thread start
- Delete the object
- Ensure thread is cleaned up

---

## Notes

- The test class uses a custom `SetUpTestSuite()` and `TearDownTestSuite()` to manage a static `QCoreApplication`, needed for Qt signal processing.
- Real hardware access is abstracted behind the `IMCP2515Controller` interface.
- Tests rely on runtime signals and thread monitoring, which requires the Qt event loop (`processEvents()`).
