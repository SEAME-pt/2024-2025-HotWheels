#!/bin/bash

# Check the status of 'mini_display_info.service' and display only the active status part
echo "Checking the status of 'mini_display_info.service'..."
sudo systemctl status mini_display_info.service | grep -A 8 'mini_display_info.service - Mini Display Info'

# Check the status of 'check_api.service' and display only the active status part
echo "Checking the status of 'check_api.service'..."
sudo systemctl status check_api.service | grep -A 8 'check_api.service - Check Github API Service'
echo "------------------------------------------------------------------------------"

# Check the status of 'car_controls.service' and display only the active status part
echo "Checking the status of 'car_controls.service'..."
sudo systemctl status car_controls.service | grep -A 8 'car_controls.service - Car Controls Service'
echo "------------------------------------------------------------------------------"

# Check the status of 'cluster.service' and display only the active status part
echo "Checking the status of 'cluster.service'..."
sudo systemctl status cluster.service | grep -A 8 'cluster.service - Cluster Service'
