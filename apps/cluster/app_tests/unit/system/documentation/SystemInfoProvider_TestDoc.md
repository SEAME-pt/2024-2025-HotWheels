# Unit Tests: SystemInfoProvider

This document provides an overview of the unit tests defined in `test_SystemInfoProvider.cpp` for the `SystemInfoProvider` class.

## Overview

These tests validate the functionality of the `SystemInfoProvider`, which is responsible for retrieving system-level data such as:
- WiFi status and network name
- Device temperature
- Local IP address

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)

Dependencies are mocked using a custom `MockSystemCommandExecutor`.

---

## ✅ Test Cases

### 1. `GetWifiStatus_Connected`
**Purpose:** Ensure proper parsing of a connected WiFi interface.
**Checks:**
- Status is `"Connected"`
- WiFi name is `"MyWiFi"`

---

### 2. `GetWifiStatus_Disconnected`
**Purpose:** Handle cases where WiFi is present but not connected.
**Checks:**
- Status is `"Disconnected"`
- WiFi name is empty

---

### 3. `GetWifiStatus_NoInterface`
**Purpose:** Handle cases where no wireless interface is present.
**Checks:**
- Status is `"No interface detected"`
- WiFi name is empty

---

### 4. `GetTemperature_ValidReading`
**Purpose:** Parse a valid temperature file reading from sysfs.
**Checks:**
- Raw `45000` converts to `"45.0°C"`

---

### 5. `GetTemperature_InvalidReading`
**Purpose:** Handle non-numeric file input.
**Checks:**
- Output is `"N/A"`

---

### 6. `GetIpAddress_Valid`
**Purpose:** Extract and return a valid IPv4 address for wlan0.
**Checks:**
- Output is `"192.168.1.100"`

---

### 7. `GetIpAddress_NoIP`
**Purpose:** Handle cases where no IP is assigned to wlan0.
**Checks:**
- Output is `"No IP address"`

---

## Notes

- Command execution and file reads are mocked to isolate the `SystemInfoProvider` from real hardware.
- Network and temperature interfaces rely on Linux utilities and sysfs paths.
- All logic falls back gracefully when encountering missing or malformed data.
