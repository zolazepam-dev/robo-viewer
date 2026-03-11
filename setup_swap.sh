#!/bin/bash
# Swap Setup Script - Run with sudo
# Usage: sudo ./setup_swap.sh

SWAP_SIZE=8G
SWAP_FILE=/swapfile

echo "🔧 Setting up ${SWAP_SIZE} swap file..."

if [ -f "$SWAP_FILE" ]; then
    echo "⚠️  Swap file already exists. Removing..."
    sudo swapoff $SWAP_FILE
    sudo rm -f $SWAP_FILE
fi

echo "1. Allocating ${SWAP_SIZE}..."
# Check if filesystem is btrfs (needs COW disabled)
FS_TYPE=$(df -T $SWAP_FILE | tail -1 | awk '{print $2}')
if [ "$FS_TYPE" = "btrfs" ]; then
    echo "   Btrfs detected - disabling COW for swap file..."
    sudo touch $SWAP_FILE
    sudo chattr +C $SWAP_FILE
    sudo dd if=/dev/zero of=$SWAP_FILE bs=1G count=${SWAP_SIZE%G} status=progress
else
    sudo fallocate -l $SWAP_FILE $SWAP_FILE
fi

echo "2. Setting permissions..."
sudo chmod 600 $SWAP_FILE

echo "3. Creating swap area..."
sudo mkswap $SWAP_FILE

echo "4. Enabling swap..."
sudo swapon $SWAP_FILE

echo "5. Making permanent..."
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab
fi

echo ""
echo "✅ Swap setup complete!"
echo "Current swap status:"
swapon --show

echo ""
echo "Memory status:"
free -h
