# HotWheels - Instrument Cluster ⏲ 🎮
Welcome to the Instrument Cluster repository of the HotWheels team! Here you'll find all the work done during the SEAM:ME Instrument Cluster project.

## Running RS485 CAN Tests

To validate RS485 CAN communication, we have added a new test file `tests/rs485_can_test.cpp`. This test file includes scenarios for sending and receiving CAN messages using RS485.

### Instructions

1. Ensure you have the necessary dependencies installed for running the tests.
2. Navigate to the root directory of the project.
3. Run the following command to execute the tests:

```sh
make test
```

### Expected Test Results

The tests will validate the following scenarios:

- Sending CAN messages
- Receiving CAN messages
- Speed updates
- RPM updates
- Signal cadence

The test results will be displayed in the terminal, indicating whether each test passed or failed.
