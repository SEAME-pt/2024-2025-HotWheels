#!/bin/bash

# Variables
BACKUP_DIR="$HOME/Documents/myScripts/backups"  # Directory where backups will be stored

# Make sure the backup directory exists
mkdir -p $BACKUP_DIR
DATE=$(date +"%Y%m%d%H")                    # Date format for unique backup names
PARTITION_TABLE_FILE="${BACKUP_DIR}/partition_table_backup_${DATE}.txt"  # Partition table backup filename
BACKUP_IMG="${BACKUP_DIR}/backup_${DATE}.img"   # Backup image filename

# List connected devices
echo "Connected devices:"
devices=($(lsblk -d -o NAME | grep -E 'sd|mmcblk'))
for i in "${!devices[@]}"; do
    echo "[$i] ${devices[$i]}"
done

# Prompt user to select a device
read -p "Enter the number corresponding to the device you want to back up: " DEVICE_INDEX
SOURCE_DEVICE="${devices[$DEVICE_INDEX]}"

# List partitions of the selected device
echo "Partitions on /dev/$SOURCE_DEVICE:"
partitions=($(lsblk -ln -o NAME | grep "^${SOURCE_DEVICE}"))
for i in "${!partitions[@]}"; do
    echo "[$i] ${partitions[$i]}"
done

# Prompt user to select a partition
read -p "Enter the number corresponding to the partition you want to back up: " PARTITION_INDEX
SOURCE_PARTITION="${partitions[$PARTITION_INDEX]}"

# Unmount all partitions of the device
echo "Unmounting all partitions of $SOURCE_DEVICE..."
sudo umount /dev/${SOURCE_DEVICE}*

# Make sure the backup directory exists
mkdir -p $BACKUP_DIR

# Step 1: Backup the partition table
echo "Backing up partition table..."
sudo sfdisk -d /dev/$SOURCE_DEVICE > $PARTITION_TABLE_FILE
echo "Partition table backup saved to $PARTITION_TABLE_FILE"

# Step 2: Backup the system (using Partclone)
echo "Starting system backup..."
sudo partclone.ext4 -c -s /dev/$SOURCE_PARTITION -o $BACKUP_IMG

# Optional: Remove backups older than 7 days
find $BACKUP_DIR -type f -name "*.img" -mtime +15 -exec rm {} \;

# Step 3: Log the completion of the backup
echo "Backup completed at $DATE now you can use the device"
