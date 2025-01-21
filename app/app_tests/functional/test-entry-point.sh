#!/bin/bash

APP="./app/build/x86_Qt5_15_2-Debug/HotWheels-app"

# Capture both stdout and stderr
OUTPUT=$($APP --test 2>&1)

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

