#!/bin/bash

# Variables
SERVICE_NAME="cluster.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
SERVICE_USER="hotweels"
SERVICE_GROUP="hotweels"
EXECUTABLE_PATH="/home/hotweels/QtAppDeploy/QtAppJetson/CarCluster/CarCluster"
DISPLAY_ENV=":0"
RUNTIME_DIR="/run/user/1000"

# Service content
SERVICE_CONTENT="[Unit]
Description=Cluster Service
After=multi-user.target graphical.target network.target display-manager.service
Requires=graphical.target

[Service]
Type=idle
ExecStart=/home/hotweels/QtAppDeploy/QtAppJetson/CarCluster/CarCluster
Restart=on-failure
RestartSec=5
User=hotweels
Group=hotweels
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=$/run/user/1000

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

