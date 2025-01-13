#!/bin/bash

# Variables
SERVICE_NAME="check_api.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
SERVICE_USER="hotweels"
SERVICE_GROUP="hotweels"
EXECUTABLE_PATH="/home/hotweels/scripts/get_artifact_carcontrols.sh"

# Service content
SERVICE_CONTENT="[Unit]
Description=Check Github API Service
After=network.target

[Service]
ExecStart=/bin/bash /home/hotweels/scripts/get_artifact_carcontrols.sh
Restart=on-failure
User=hotweels
Group=hotweels

[Install]
WantedBy=multi-user.target"

# Check if service already exists
if [ -f "$SERVICE_PATH" ]; then
    echo "The service $SERVICE_NAME already exists at $SERVICE_PATH."
    exit 0
fi

# Step 1: Create the service file
echo "Creating service file at $SERVICE_PATH..."
echo "$SERVICE_CONTENT" | sudo tee $SERVICE_PATH > /dev/null

# Step 2: Set file permissions
echo "Setting permissions for $SERVICE_PATH..."
sudo chmod 644 $SERVICE_PATH

# Step 3: Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Step 4: Enable the service
echo "Enabling the $SERVICE_NAME service to start on boot..."
sudo systemctl enable $SERVICE_NAME

# Step 5: Start the service
echo "Starting the $SERVICE_NAME service..."
sudo systemctl start $SERVICE_NAME

echo "Done! The $SERVICE_NAME service is now set up and running."
