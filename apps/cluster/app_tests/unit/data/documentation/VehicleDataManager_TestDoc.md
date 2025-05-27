# Unit Tests: VehicleDataManager

This document describes the unit tests defined in `test_VehicleDataManager.cpp` for the `VehicleDataManager` class, responsible for processing and emitting vehicle-related data signals.

## 🧪 Overview

These tests verify:
- Signal emission for RPM, speed, mileage, direction, and steering
- Prevention of redundant signal emissions when values remain unchanged

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Qt QSignalSpy](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `RpmDataEmitsSignal`
**Purpose:** Ensure that `canDataProcessed` is emitted when RPM changes.
**Checks:**
- Correct RPM value is emitted
- Speed remains at default (0.0)

---

### 2. `SpeedDataEmitsSignalInKilometers`
**Purpose:** Ensure that speed updates trigger `canDataProcessed`.
**Checks:**
- Speed value matches expected
- RPM remains at default (0)

---

### 3. `MileageDataEmitsSignalOnChange`
**Purpose:** Ensure that `mileageUpdated` is emitted only when mileage changes.
**Steps:**
- Send a mileage update and verify signal emission
- Send same mileage again and confirm no redundant emission

---

### 4. `DirectionDataEmitsSignalOnChange`
**Purpose:** Ensure that direction changes emit `engineDataProcessed`.
**Steps:**
- Send a direction change (e.g., `Drive`) and verify signal emission
- Re-send the same direction and confirm no signal is emitted again

---

### 5. `SteeringDataEmitsSignalOnChange`
**Purpose:** Ensure that steering angle changes emit `engineDataProcessed`.
**Steps:**
- Send a steering angle and verify signal emission
- Send the same angle again and ensure no redundant emission

---

## 🛠 Notes

- This suite ensures optimal signal efficiency, avoiding unnecessary UI updates.
- Tests rely on `QSignalSpy` to introspect Qt signal payloads.
- Consistent with Qt best practices for reactive UI updates.
