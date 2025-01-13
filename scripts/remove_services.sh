#/bin/bash

echo "Stopping the services..."
sudo systemctl stop mini_display_info.service
sudo systemctl stop check_api.service
sudo systemctl stop car_controls.service
sudo systemctl stop cluser.service

echo "Disabling the services..."
sudo systemctl disable mini_display_info.service
sudo systemctl disable check_api.service
sudo systemctl disable car_controls.service
sudo systemctl disable cluser.service

echo "Removing the config file from the services..."
sudo rm /etc/systemd/system/mini_display_info.service
sudo rm /etc/systemd/system/check_api.service
sudo rm /etc/systemd/system/car_controls.service
sudo rm /etc/systemd/system/cluser.service

echo "Reload the system..."
sudo systemctl daemon-reload

echo "Verifiying they are erased..."
sudo systemctl status mini_display_info.service
sudo systemctl status check_api.service
sudo systemctl status car_controls.service
sudo systemctl status cluser.service

