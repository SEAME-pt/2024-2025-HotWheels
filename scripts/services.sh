#!/bin/bash

# Check the status of 'check_api.service' and display only the active status part
echo "Checking the status of 'check_api.service'..."
sudo systemctl status check_api.service | grep -A 8 'check_api.service - Check Github API Service'

# Check the status of 'car_controls.service' and display only the active status part
echo "Checking the status of 'car_controls.service'..."
sudo systemctl status car_controls.service | grep -A 8 'car_controls.service - Car Controls Service'
