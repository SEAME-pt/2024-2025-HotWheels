# Unit Tests: SystemManager

This document provides an overview of the unit tests defined in `test_SystemManager.cpp` for the `SystemManager` class.

## Overview

These tests validate the behavior of `SystemManager`, which is responsible for aggregating and emitting system-related signals such as:
- Date and time
- WiFi status
- Temperature
- Battery percentage

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)
- [Qt Test Utilities](https://doc.qt.io/qt-6/qsignalspy.html)

Mock dependencies:
- `MockSystemInfoProvider`
- `MockBatteryController`

---

## ✅ Test Cases

### 1. `UpdateTime_EmitsCorrectSignal`
**Purpose:** Ensure that the `SystemManager` emits the `timeUpdated` signal with valid values.
**Checks:**
- Date, time, and day strings are non-empty.

---

### 2. `UpdateSystemStatus_EmitsWifiStatus`
**Purpose:** Validate that `wifiStatusUpdated` is emitted with correct values from the mock provider.
**Setup:**
- Mock returns `Connected` with WiFi name `"MyWiFi"`.
**Checks:**
- Signal is emitted with both expected values.

---

### 3. `UpdateSystemStatus_EmitsTemperature`
**Purpose:** Verify that the system emits the current temperature value.
**Setup:**
- Mock returns `"42.0°C"` for the temperature.
**Checks:**
- Signal is emitted with the expected temperature.

---

### 4. `UpdateSystemStatus_EmitsBatteryPercentage`
**Purpose:** Confirm that `batteryPercentageUpdated` is emitted with the correct value.
**Setup:**
- Mock battery controller returns `75.0f`.
**Checks:**
- Emitted value matches the expected percentage.

---

## Notes

- `QSignalSpy` is used in all tests to capture and inspect emitted signals.
- Each signal is emitted as a result of calling the appropriate update method on the `SystemManager`.
- Dependencies are injected through mocks to isolate unit behavior.
