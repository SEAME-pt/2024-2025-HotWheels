# Unit Tests: MileageManager

This document provides an overview of the unit tests defined in `test_MileageManager.cpp` for the `MileageManager` class.

## 🧪 Overview

These tests validate:
- Initialization behavior and data loading
- Proper communication with `MileageCalculator` and `MileageFileHandler`
- Emission of update signals
- Persistence of mileage state

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)
- Qt `QSignalSpy` for signal verification

---

## ✅ Test Cases

### 1. `Initialize_LoadsMileageFromFile`
**Purpose:** Ensure that the stored mileage is loaded from the file on initialization.
**How:**
- `readMileage()` mocked to return `123.45`
- `writeMileage()` expected to be called during shutdown

---

### 2. `OnSpeedUpdated_CallsCalculator`
**Purpose:** Verify that incoming speed updates are passed to the mileage calculator.
**How:**
- `addSpeed(50.0)` is expected to be called once

---

### 3. `UpdateMileage_EmitsMileageUpdatedSignal`
**Purpose:** Ensure that updated mileage is emitted via the `mileageUpdated` signal.
**How:**
- `calculateDistance()` returns `10.5`
- A `QSignalSpy` confirms emission with expected value

---

### 4. `SaveMileage_CallsFileHandler`
**Purpose:** Confirm that mileage is correctly saved to the file.
**How:**
- Two mileage updates are triggered (50.0 + 150.0)
- `writeMileage(200.0)` is expected

---

## 🛠 Notes

- Mocks are injected through constructor dependency injection.
- The `NiceMock` wrapper is used to ignore uninteresting calls.
- The mileage manager accumulates calculated mileage internally, not resetting the calculator on each update.
