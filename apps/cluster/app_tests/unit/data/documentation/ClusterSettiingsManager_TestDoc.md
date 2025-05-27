# Unit Tests: ClusterSettingsManager

This document provides an overview of the unit tests defined in `test_ClusterSettingsManager.cpp` for the `ClusterSettingsManager` class, which handles user-configurable cluster settings such as driving mode, theme, and metric units.

## 🧪 Overview

These tests validate:
- Proper toggling logic for driving mode, cluster theme, and metric units
- Emission of corresponding Qt signals upon changes

Test frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Qt QSignalSpy](https://doc.qt.io/qt-6/qsignalspy.html) for signal monitoring

---

## ✅ Test Cases

### 1. `ToggleDrivingModeEmitsSignal`
**Purpose:** Ensure `toggleDrivingMode()` switches between `Manual` and `Automatic` modes.
**How:** Calls the method twice and inspects `drivingModeUpdated` signal emissions using `QSignalSpy`.

---

### 2. `ToggleClusterThemeEmitsSignal`
**Purpose:** Ensure `toggleClusterTheme()` switches between `Dark` and `Light` themes.
**How:** Monitors `clusterThemeUpdated` signal with `QSignalSpy`.

---

### 3. `ToggleClusterMetricsEmitsSignal`
**Purpose:** Ensure `toggleClusterMetrics()` toggles between `Kilometers` and `Miles`.
**How:** Uses `QSignalSpy` to track the `clusterMetricsUpdated` signal.

---

## 🛠 Notes

- These tests assume default initial values:
  - Driving Mode: `Manual`
  - Cluster Theme: `Dark`
  - Cluster Metrics: `Kilometers`
- All state toggles are verified through signal-based state observation, ensuring integration with UI-bound logic.

