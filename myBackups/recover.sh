#!/bin/bash

# Variables
BACKUP_DIR="$HOME/Documents/myScripts/backups"  # Directory where backups are stored

# List connected devices
echo "Connected devices:"
devices=($(lsblk -d -o NAME | grep -E 'sd|mmcblk'))
for i in "${!devices[@]}"; do
    echo "[$i] ${devices[$i]}"
done

# Prompt user to select a device
read -p "Enter the number corresponding to the device you want to recover: " DEVICE_INDEX
DEVICE="${devices[$DEVICE_INDEX]}"

# List partitions of the selected device
echo "Partitions on /dev/$DEVICE:"
partitions=($(lsblk -ln -o NAME | grep "^${DEVICE}"))
for i in "${!partitions[@]}"; do
    echo "[$i] ${partitions[$i]}"
done

# Prompt user to select a partition
read -p "Enter the number corresponding to the partition you want to recover: " PARTITION_INDEX
PARTITION="${partitions[$PARTITION_INDEX]}"

# List available backup images
echo "Available backup images:"
backup_images=($(ls $BACKUP_DIR/backup_*.img))
for i in "${!backup_images[@]}"; do
    echo "[$i] ${backup_images[$i]}"
done

# Prompt user to select a backup image
read -p "Enter the number corresponding to the backup image you want to use: " BACKUP_INDEX
BACKUP_IMG="${backup_images[$BACKUP_INDEX]}"
PARTITION_TABLE_FILE="${BACKUP_IMG/backup_/partition_table_backup_}"
PARTITION_TABLE_FILE="${PARTITION_TABLE_FILE/.img/.txt}"

# Check if the partition table file exists
if [ ! -f "$PARTITION_TABLE_FILE" ]; then
    echo "Partition table file $PARTITION_TABLE_FILE not found!"
    exit 1
fi

# Step 1: Unmount the device if it is mounted
echo "Unmounting $DEVICE if it is mounted..."
sudo umount /dev/${DEVICE}* 2>/dev/null

# Step 2: Restore the partition table
echo "Restoring partition table from $PARTITION_TABLE_FILE to $DEVICE..."
sudo sfdisk /dev/$DEVICE < $PARTITION_TABLE_FILE

# Step 3: Check and resize the filesystem
echo "Checking filesystem on /dev/${PARTITION}..."
sudo e2fsck -f /dev/${PARTITION}

echo "Resizing partition /dev/${PARTITION} to fit the target device..."
sudo resize2fs /dev/${PARTITION}

# Step 4: Restore the backup image
echo "Restoring backup image $BACKUP_IMG to /dev/${PARTITION}..."
sudo partclone.ext4 -r -C -s $BACKUP_IMG -o /dev/${PARTITION}

# Step 5: Log the completion of the recovery
echo "Recovery completed. The device $DEVICE has been restored."