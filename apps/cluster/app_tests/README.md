# Application Test Suite

This directory contains all automated tests for the application, categorized to support clarity, scalability, and maintainability.

---

## 📁 Directory Structure

app_tests/
├── unit/           # Tests for individual classes or functions
├── integration/    # Tests for interactions between components or modules
├── mocks/          # Mock files for simulating dependencies
├── functional/     # High-level tests for application-wide behavior

---

## Testing Categories

### 1. Unit Tests (`unit/`)

- Test individual functions or classes in isolation.
- Dependencies are mocked to avoid side effects.
- Typically written using [Google Test](https://github.com/google/googletest) and [Google Mock](https://github.com/google/googletest/tree/main/googlemock).

### 2. Integration Tests (`integration/`)

- Verify correct interactions between multiple real components.
- Help detect interface mismatches, threading issues, and signal/slot integration (especially with Qt).

### 3. Functional Tests (`functional/`)

- Validate the application from a black-box perspective.
- Focus on user workflows and overall system functionality.
- Includes test scripts like `test_entry_point.sh`.

### 4. Mock Implementations (`mocks/`)

- Provide test doubles for system-level or external dependencies.
- Used primarily in unit tests to isolate logic under test.

---

## Test Documentation

Every test file is accompanied by a `.md` file that documents:

- What the test covers
- How it works internally
- Related signals, slots, or classes
- Edge cases or failure expectations
