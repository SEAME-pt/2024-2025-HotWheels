app_tests

This directory contains tests for the application, categorized to ensure clarity and maintainability.
Directory Structure

app_tests/
├── unit/           # Tests for individual classes or functions
├── integration/    # Tests for interactions between components or modules
├── mocks/          # Mock files for simulating dependencies
├── functional/     # High-level tests for application-wide behavior

Testing Categories
1. Unit Tests

    Focus on testing individual classes or functions in isolation.
    Mock dependencies when required.

2. Integration Tests

    Verify that multiple components work together as expected.
    Typically involve real implementations of the components.

3. Mock Files

    Provide mock classes or functions for isolating dependencies in unit tests.

4. Functional Tests

    Test high-level application behavior.
    Includes scripts like test_entry_point.sh for validating overall functionality.
