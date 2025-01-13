#!/bin/bash

# Variables
SERVICE_NAME="mini_display_info.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
SERVICE_USER="hotweels"
SERVICE_GROUP="hotweels"

# Service content
SERVICE_CONTENT="[Unit]
Description=Mini Display Info
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/hotweels/Documents/UPS-Power-Module/ups_display/display_server.py
Restart=always
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
