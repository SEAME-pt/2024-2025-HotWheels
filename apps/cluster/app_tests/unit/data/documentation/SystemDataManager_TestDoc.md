# Unit Tests: SystemDataManager

This document outlines the unit tests implemented in `test_SystemDataManager.cpp` for the `SystemDataManager` class, which handles system-related UI data such as time, temperature, and battery percentage.

## Overview

These tests ensure:
- Correct signal emission when data changes
- Signal suppression when unchanged data is received

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Qt QSignalSpy](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `TimeDataEmitsSignal`
**Purpose:** Ensure `systemTimeUpdated` signal is emitted with the correct values.
**How:** Call `handleTimeData()` and verify signal contents with `QSignalSpy`.

---

### 2. `TemperatureDataEmitsSignalOnChange`
**Purpose:** Confirm that `systemTemperatureUpdated` emits only on new temperature values.
**How:**
- Call `handleTemperatureData()` with a value and verify signal emission
- Re-call with same value and check no signal is emitted

---

### 3. `BatteryPercentageEmitsSignalOnChange`
**Purpose:** Ensure `batteryPercentageUpdated` is emitted only when the percentage changes.
**How:**
- Call `handleBatteryPercentage()` with a float
- Call again with same value and confirm no new signal is emitted

---

## Notes

- These tests help prevent unnecessary UI updates by ensuring signal emission occurs only when values actually change.
- Tests rely on `QSignalSpy` to observe emitted Qt signals in real-time.
