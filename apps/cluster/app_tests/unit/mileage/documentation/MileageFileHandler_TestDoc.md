# Unit Tests: MileageFileHandler

This document provides an overview of the unit tests defined in `test_MileageFileHandler.cpp` for the `MileageFileHandler` class.

## 🧪 Overview

These tests validate:
- File existence and creation logic
- Reading and writing mileage data from/to the file
- Handling of invalid or missing file content gracefully

Frameworks used:
- [Google Test](https://github.com/google/googletest)
- [Google Mock](https://github.com/google/googletest/tree/main/googlemock)

---

## ✅ Test Cases

### 1. `EnsureFileExists_FileDoesNotExist_CreatesFileWithZeroMileage`
**Purpose:** Ensure the file is created and initialized with `"0.0"` when it does not exist.
**Mocks:**
- `exists(path)` returns `false`
- `open()` and `write()` are expected to be called

---

### 2. `ReadMileage_ValidNumber_ReturnsParsedValue`
**Purpose:** Ensure mileage is correctly read from the file when a valid float is stored.
**Mocks:**
- Simulates a file returning the string `"123.45"`
**Expect:** Returned value is `123.45`

---

### 3. `ReadMileage_InvalidNumber_ReturnsZero`
**Purpose:** Ensure invalid file content is safely handled.
**Mocks:**
- Simulates a file returning the string `"INVALID"`
**Expect:** Returned value is `0.0`

---

### 4. `WriteMileage_ValidNumber_WritesToFile`
**Purpose:** Ensure mileage value is written to the file correctly.
**Mocks:**
- `open()` and `write()` should be called with `"789.12"`

---

## 🛠 Notes

- `MileageFileHandler` is injected with custom lambdas for file operations, allowing complete control via `MockFileController`.
- File I/O is mocked entirely to isolate and test logic without real file system dependencies.
- The use of dependency injection (DI) enhances testability and separation of concerns.
