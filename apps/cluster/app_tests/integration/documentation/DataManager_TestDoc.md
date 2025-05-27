# Integration Tests: DataManager

This document outlines the integration tests defined in `test_int_DataManager.cpp` for the `DataManager` class, which orchestrates data flow between system components like `SystemDataManager`, `VehicleDataManager`, and `ClusterSettingsManager`.

## 🧪 Overview

These tests verify:
- Correct forwarding of CAN and engine-related data
- Emission of UI-related update signals (time, temperature, battery)
- Functionality of toggling states like driving mode, theme, and metrics

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Qt Signal Spy](https://doc.qt.io/qt-6/qsignalspy.html)

---

## ✅ Test Cases

### 1. `ForwardSpeedDataToVehicleDataManager`
**Purpose:** Ensure speed data is forwarded via `canDataProcessed`.
**How:** Calls `handleSpeedData` and verifies signal emission.

### 2. `ForwardRpmDataToVehicleDataManager`
**Purpose:** Ensure RPM data is correctly propagated.
**How:** Emits RPM with `handleRpmData` and checks `canDataProcessed`.

### 3. `ForwardSteeringDataToVehicleDataManager`
**Purpose:** Ensure steering data updates are emitted via `engineDataProcessed`.
**How:** Sends a sample steering value and verifies signal content.

### 4. `ForwardDirectionDataToVehicleDataManager`
**Purpose:** Check if direction enum is forwarded.
**How:** Sends `CarDirection::Drive` and inspects emitted signal.

### 5. `ForwardTimeDataToSystemDataManager`
**Purpose:** Ensure UI time updates are correctly handled.
**How:** Sends date, time, and weekday to `handleTimeData`.

### 6. `ForwardTemperatureDataToSystemDataManager`
**Purpose:** Forward temperature strings from backend to UI.
**How:** Tests `handleTemperatureData` and verifies emitted text.

### 7. `ForwardBatteryPercentageToSystemDataManager`
**Purpose:** Validate float percentage forwarding to UI.
**How:** Calls `handleBatteryPercentage` and expects correct float.

### 8. `ForwardMileageUpdateToVehicleDataManager`
**Purpose:** Ensure mileage values reach the UI and subsystems.
**How:** Uses `handleMileageUpdate` and validates emission.

### 9. `ToggleDrivingMode`
**Purpose:** Test state toggling via UI or logic triggers.
**How:** Calls `toggleDrivingMode` and expects `drivingModeUpdated`.

### 10. `ToggleClusterTheme`
**Purpose:** Ensure theme changes trigger appropriate signals.
**How:** Invokes `toggleClusterTheme`.

### 11. `ToggleClusterMetrics`
**Purpose:** Validate toggling of metric display logic.
**How:** Invokes `toggleClusterMetrics` and checks signal.

---

## 🛠 Notes

- A static `QCoreApplication` is used for enabling Qt event loop functionality (`processEvents()`).
- Tests use `QSignalSpy` to capture and analyze emitted Qt signals.
- All subsystems (`SystemDataManager`, `VehicleDataManager`, etc.) are initialized via the real `DataManager`, not mocks.
- Focus is on integration through signal flow rather than internal logic or hardware behavior.
