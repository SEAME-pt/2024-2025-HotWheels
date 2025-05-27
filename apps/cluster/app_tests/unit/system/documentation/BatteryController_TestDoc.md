# Unit Tests: BatteryController

This document provides an overview of the unit tests defined in `test_BatteryController.cpp` for the `BatteryController` class.

## 🧪 Overview

These tests validate:
- Proper initialization behavior
- Accurate computation of battery percentage based on I2C sensor data

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)

---

## ✅ Test Cases

### 1. `Initialization_CallsCalibration`
**Purpose:** Ensure that calibration is performed during controller construction.
**How:**
- Expects a write to I2C register `0x05` with the value `4096`
- Verifies calibration logic is triggered on creation

---

### 2. `GetBatteryPercentage_CorrectCalculation`
**Purpose:** Validate the computation of battery percentage using raw voltage values.
**How:**
- Mocks raw register reads: `bus voltage = 1000`, `shunt voltage = 100`
- Internally calculates:
  - `busVoltage = (1000 >> 3) * 0.004`
  - `shuntVoltage = 100 * 0.01`
  - `loadVoltage = busVoltage + shuntVoltage`
  - `percentage = (loadVoltage - 6.0) / 2.4 * 100`
- Caps percentage within 0–100%
- Verifies returned value matches expected within a 0.1f margin

---

## 🛠 Notes

- The test class uses `NiceMock` to suppress uninteresting mock warnings.
- `BatteryController` depends on an `I2CController` interface to abstract hardware.
- Voltage constants (e.g., 6.0V min, 2.4V range) are hardcoded into the formula, reflecting battery-specific assumptions.
