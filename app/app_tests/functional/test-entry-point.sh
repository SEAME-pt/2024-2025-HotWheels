#!/bin/bash
# -*- tab-width: 4; encoding: utf-8 -*-
#
## @file test-entry-point.sh
## @brief Test the entry point of the application.
## @author @Fle-bihh (original author)
## @author @HotWheels (adapted)
## @date 2025-01-30
## @version 1.0
## @copyright MIT License
## @details This script tests the application's entry point by running it in test mode
## and comparing the output with the expected result.
##
## @section Usage
## Run this script in a terminal:
## ```
## ./test-entry-point.sh
## ```
## The script will compare the output of the application and return a success or failure message.
##
## @return 0 if the test passes, 1 if it fails.

## The path to the application
APP="./app/build/x86_Qt5_15_2-Debug/HotWheels-app"

## Capture both stdout and stderr
OUTPUT=$($APP --test 2>&1)

## The expected output of the application
EXPECTED_OUTPUT="[Main] HotWheels Cluster starting...
[Main] Test mode activated. Exiting..."

# Trim whitespace for both outputs
OUTPUT=$(echo "$OUTPUT" | xargs)
EXPECTED_OUTPUT=$(echo "$EXPECTED_OUTPUT" | xargs)

if [ "$OUTPUT" == "$EXPECTED_OUTPUT" ]; then
    echo "Test passed!"
    exit 0
else
    echo "Test failed!"
    echo "Expected:"
    echo "[$EXPECTED_OUTPUT]"
    echo "Got:"
    echo "[$OUTPUT]"
    exit 1
fi
