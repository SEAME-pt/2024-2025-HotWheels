
#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <partition_table_file.txt> <device>"
    exit 1
fi

PARTITION_TABLE_FILE="$1"  # Partition table file
DEVICE="$2"                # Device to recover

# Step 1: Restore the partition table
echo "Restoring partition table from $PARTITION_TABLE_FILE to $DEVICE..."
sudo sfdisk /dev/$DEVICE < $PARTITION_TABLE_FILE

# Step 2: Restore the backup image
BACKUP_IMG="${PARTITION_TABLE_FILE/partition_table_backup_/backup_}"
BACKUP_IMG="${BACKUP_IMG/.txt/.img}"

if [ ! -f "$BACKUP_IMG" ]; then
    echo "Backup image $BACKUP_IMG not found!"
    exit 1
fi

echo "Restoring backup image $BACKUP_IMG to ${DEVICE}1..."
sudo partclone.ext4 -r -s $BACKUP_IMG -o /dev/${DEVICE}1

# Step 3: Log the completion of the recovery
echo "Recovery completed. The device $DEVICE has been restored."