#!/bin/bash

JETSON=jetson@hotwheels-car.netbird.cloud

# Sync sysroot from the Jetson
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/lib sysroot
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/include sysroot/usr
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/lib sysroot/usr
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/src sysroot/usr
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/local/lib sysroot/usr/local
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/opt/nvidia sysroot/opt
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/local/cuda-10.2 sysroot/usr/local
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/local/include sysroot/usr/local
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/local/include/GL sysroot/usr/local/include
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/lib/aarch64-linux-gnu sysroot/usr/lib/aarch64-linux-gnu
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/etc/alternatives/ sysroot/etc/alternatives

# Create CUDA symlink to match Jetson's /usr/local/cuda -> cuda-10.2
cd sysroot/usr/local
if [ -d cuda-10.2 ] && [ ! -e cuda ]; then
    ln -sf cuda-10.2 cuda
    echo "Created CUDA symlink: cuda -> cuda-10.2"
fi
cd - > /dev/null

# Additional essential directories for Qt cross-compilation
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/share/pkgconfig sysroot/usr/share
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/lib/pkgconfig sysroot/usr/lib
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/lib/aarch64-linux-gnu/pkgconfig sysroot/usr/lib/aarch64-linux-gnu
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/lib/aarch64-linux-gnu sysroot/lib
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/share/cmake sysroot/usr/share
rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/usr/lib/cmake sysroot/usr/lib

./sysroot-relativelinks.py sysroot
