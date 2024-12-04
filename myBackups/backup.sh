#!/bin/bash

# Variables
BACKUP_DIR="/home/michel/Documents/myScripts/backups"  # Directory where backups will be stored
DATE=$(date +"%Y%m%d%H")                    # Date format for unique backup names
PARTITION_TABLE_FILE="${BACKUP_DIR}/partition_table_backup_${DATE}.txt"  # Partition table backup filename
BACKUP_IMG="${BACKUP_DIR}/backup_${DATE}.img"   # Backup image filename

# Partition to be backed up (adjust if needed)
if [ -z "$1" ]; then
	echo "Usage: $0 <source_device>"
	exit 1
fi

SOURCE_DEVICE="$1"                         # Partition or disk to back up

# Unmount all partitions of the device
echo "Unmounting all partitions of $SOURCE_DEVICE..."
sudo umount /dev/${SOURCE_DEVICE}*

SOURCE_PARTITION="${SOURCE_DEVICE}1"                   # Partition or disk to back up
PARTITION_TYPE="ext4"                           # Filesystem type (adjust accordingly)

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
find $BACKUP_DIR -type f -name "*.img" -mtime +7 -exec rm {} \;

# Step 3: Log the completion of the backup
echo "Backup completed at $DATE now you can use the device"
