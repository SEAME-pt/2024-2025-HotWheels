### Build arm64 Qt in order to cross-compile Qt app for Jetson Nano (arm64 - Ubuntu 18)

Follow this tutorial with the following adaptations : 
https://www.stefanocottafavi.com/crosscompile-qt-for-rpi/

 - If a step is not mentionned in this file, assume you need to do it.

# On Target

 - Start after the raspebrry pi flashing step.

 - Install more libraries on target to make sure XCB / EGLFS plugins will be built with Qt. ( -- The following list might not be enough -- )
 
		sudo apt-get install \
		"libx11-*" \
		"libx11*" \
		"libxcb-*" \
		"libxcb*"

 - Symlinks step: Use /usr/include/aarch64-linux-gnu instead of /usr/include/arm-linux-gnueabihf

 - Set up ssh keys - it's optional but huge gain of time for future steps

# On Host

 - To fix certain errors when compiling Qt, I recommend using g++ v9 (g++ -v / ChatGPT knows how to change your computer version)

 - Adapt folder naming for Jetson instead of Pi ( -- Clarity -- ), and consider exporting a JETSON variable for ssh address (in .zshrc for example:
    export JETSON="hotweels@10.21.221.78")

 - rsync -avz --rsync-path="sudo rsync" --delete $JETSON:/opt/vc sysroot/opt
    --> /opt/vc equivalent folder on Jetson is /opt/nvidia (Copy this one instead)

 - adjust compiler step: Do not use the cp and sed commands, instead modify the following files (home/seame/qtjetson --> your/path/to/qt/build/folder) :
 
    -- qt-everywhere-src-5.15.2/qtbase/mkspecs/linux-aarch64-gnu-g++/qmake.conf --

        #
        # qmake configuration for building with aarch64-linux-gnu-g++
        #

        MAKEFILE_GENERATOR      = UNIX
        CONFIG                 += incremental
        QMAKE_INCREMENTAL_STYLE = sublib

        include(../common/linux.conf)
        include(../common/gcc-base-unix.conf)
        include(../common/g++-unix.conf)

        # modifications to g++.conf
        QMAKE_CC                = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc
        QMAKE_CXX               = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++
        QMAKE_LINK              = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++
        QMAKE_LINK_SHLIB        = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++

        # modifications to linux.conf
        QMAKE_AR                = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-ar cqs
        QMAKE_OBJCOPY           = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-objcopy
        QMAKE_NM                = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-nm -P
        QMAKE_STRIP             = /home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-strip
        load(qt_config)

    -- qt-everywhere-src-5.15.2/qtbase/mkspecs/devices/linux-jetson-tk1-g++/qmake.conf  --    

        #
        # qmake configuration for the Jetson Nano boards running Linux For Tegra
        #
        # Note that this configuration assumes cross-compilation tools are specified in the configure step.
        #
        DISTRO_OPTS += aarch64

        include(../common/linux_device_pre.conf)

        # Include and library paths for Jetson Nano sysroot
        QMAKE_INCDIR_POST += \
            $$[QT_SYSROOT]/usr/include \
            $$[QT_SYSROOT]/usr/include/aarch64-linux-gnu

        QMAKE_LIBDIR_POST += \
            $$[QT_SYSROOT]/usr/lib \
            $$[QT_SYSROOT]/lib/aarch64-linux-gnu \
            $$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu

        QMAKE_RPATHLINKDIR_POST += \
            $$[QT_SYSROOT]/usr/lib \
            $$[QT_SYSROOT]/usr/lib/aarch64-linux-gnu \
            $$[QT_SYSROOT]/lib/aarch64-linux-gnu

        # Architecture-specific flags for AArch64 (Jetson Nano)
        COMPILER_FLAGS          += -march=armv8-a

        # EGLFS integration for Jetson Nano
        EGLFS_DEVICE_INTEGRATION = eglfs_kms_egldevice

        include(../common/linux_arm_device_post.conf)
        load(qt_config)

 - Use the following configure command inside /qtjetson/build folder :
 
            ../qt-everywhere-src-5.15.2/configure \
            -release \
            -opengl es2 \
            -eglfs \
            -xcb \
            -xcb-xlib \
            -platform linux-g++ \
            -device linux-jetson-tk1-g++ \
            -device-option CROSS_COMPILE=/home/seame/qtjetson/tools/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu- \
            -sysroot /home/seame/qtjetson/sysroot \
            -prefix /usr/local/qt5.15 \
            -extprefix /home/seame/qtjetson/qt5.15 \
            -opensource \
            -confirm-license \
            -skip qtscript \
            -skip qtwayland \
            -skip qtwebengine \
            -nomake tests \
            -make libs \
            -pkg-config \
            -no-use-gold-linker \
            -v \
            -L$HOME/qtjetson/sysroot/usr/lib/aarch64-linux-gnu \
            -I$HOME/qtjetson/sysroot/usr/include/aarch64-linux-gnu


    --> Do not make or make install until you see the following "yes"s in config.summary (Plugins required to display Qt app). If you have unexpected "no"s, you have missing libraries in the target sysroot (not on host). So you need to install them there, re-sync and re-link the sysroot on host and configure again :

        EGL-X11 Plugin ..................... yes
        GLib ................................... yes
        EGL .................................... yes
        OpenGL:
            OpenGL ES 2.0 ........................ yes
            OpenGL ES 3.0 ........................ yes
            OpenGL ES 3.1 ........................ yes
            OpenGL ES 3.2 ........................ yes
        xkbcommon .............................. yes
        X11 specific:
            XLib ................................. yes
            XCB Xlib ............................. yes
            EGL on X11 ........................... yes
            xkbcommon-x11 ........................ yes
        QPA backends:
        EGLFS .................................. yes
        EGLFS details:
            EGLFS X11 ............................ yes
        LinuxFB ................................ yes
        VNC .................................... yes
        XCB:
            EGL-X11 Plugin ..................... yes

# Build and Deploy

 - "Configure Qt Creator" and "Configure Qt project" steps are not required, you can now build and deploy from terminal.

 - /path/to/arm64/qmake YourRootFile.pro (/home/seame/qtjetson/qt5.15/bin/qmake ../CarClusterRoot.pro in my case)
 - make 
 - file executable --> You should see arm64 informations
 - rsync your executable on target and run it!

# Debug

 - QQuickView didn't work, had to use QWidget instead to contain my Qml Module.
