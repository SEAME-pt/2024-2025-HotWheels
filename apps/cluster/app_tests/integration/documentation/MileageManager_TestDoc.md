# Integration Tests: MileageManager

This document summarizes the integration tests in `test_int_MileageManager.cpp` for the `MileageManager` class, which is responsible for tracking and managing mileage updates using the `MileageCalculator` and `MileageFileHandler`.

## 🧪 Overview

These tests verify:
- Signal emissions on state updates
- Proper integration with calculator and file handler
- Basic lifecycle behavior (initialize, save, shutdown)

Test framework used:
- [Google Test](https://github.com/google/googletest)
- [Qt Signal Spy](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `ForwardMileageData`
**Purpose:** Ensure the `mileageUpdated` signal is emitted after a speed update.
**How:** Call `onSpeedUpdated()` and listen with `QSignalSpy`.

### 2. `InitializeMileageManager`
**Purpose:** Verify that initializing the manager triggers a mileage signal.
**How:** Call `initialize()` and expect `mileageUpdated` with value `0.0`.

### 3. `UpdateMileageOnSpeedUpdate`
**Purpose:** Validate that calling `onSpeedUpdated()` updates mileage and emits a signal.
**How:** Same as test 1, checks correct mileage in emitted signal.

### 4. `SaveMileage`
**Purpose:** Test if mileage is correctly saved and retrievable.
**How:** Simulate speed update, call `saveMileage()`, check `getTotalMileage()`.

### 5. `UpdateTimerInterval`
**Purpose:** Confirm that the timer interval emits a mileage update over time.
**How:** Call `initialize()` and wait for a `QTimer` event to trigger a signal.

### 6. `ShutdownMileageManager`
**Purpose:** Validate shutdown clears or resets internal mileage state.
**How:** Call `shutdown()` and expect total mileage to be reset to `0.0`.

---

## 🛠 Notes

- The test fixture allocates `MileageCalculator` and `MileageFileHandler` directly.
- Uses a local test JSON file path for file-backed mileage persistence.
- `QCoreApplication` is used to enable signal/slot processing.
- Tests rely heavily on `QSignalSpy` to validate that state changes produce Qt signals.
