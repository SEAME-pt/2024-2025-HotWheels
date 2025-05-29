# Integration Tests: SystemManager

This document outlines the integration tests found in `test_int_SystemManager.cpp` for the `SystemManager` class, which integrates battery monitoring, system information, and system command execution.

## Overview

These tests verify:
- Correct signal emission from system monitoring
- Proper initialization and shutdown behaviors
- Integration with `BatteryController`, `SystemInfoProvider`, and `SystemCommandExecutor`

Test framework used:
- [Google Test](https://github.com/google/googletest)
- [Qt SignalSpy](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `UpdateTimeSignal`
**Purpose:** Validate that `SystemManager` emits the `timeUpdated` signal on `initialize()`.
**Check:** The signal is emitted and not empty.

### 2. `UpdateWifiStatusSignal`
**Purpose:** Ensure `wifiStatusUpdated` is emitted when `initialize()` is called.
**Check:** Signal is captured with valid arguments.

### 3. `UpdateTemperatureSignal`
**Purpose:** Ensure `temperatureUpdated` is emitted.
**Check:** Argument list is not empty after initialization.

### 4. `UpdateBatteryPercentageSignal`
**Purpose:** Validate battery percentage signal range.
**Checks:**
- Signal is emitted
- Value is between 0 and 100

### 5. `ShutdownSystemManager`
**Purpose:** Ensure shutdown deactivates timers.
**Check:** `timeTimer` and `statusTimer` are no longer active after `shutdown()`.

---

## Notes

- The test suite uses a real instance of `BatteryController`, `SystemInfoProvider`, and `SystemCommandExecutor` (not mocked).
- A static `QCoreApplication` is created to allow Qt signals to function properly.
- The use of `QSignalSpy` enables detailed introspection of signal behavior.
