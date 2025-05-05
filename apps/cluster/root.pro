QT       += core gui widgets network
CONFIG   += c++17

# Base directory (absolute path to the root of the project)
BASE_DIR = $$PWD

TEMPLATE = subdirs

# Always include the main app
SUBDIRS += app_target
app_target.file = $$BASE_DIR/HotWheels-app.pro

# Only include test targets when not cross-compiling for ARM
!contains(QT_ARCH, "arm") & \
!contains(QT_ARCH, "arm64") & \
!contains(QT_ARCH, "aarch64") {

    SUBDIRS += unit_tests_target \
               integration_tests_target

    unit_tests_target.file = $$BASE_DIR/HotWheels-unit-tests.pro
    unit_tests_target.depends = app_target

    integration_tests_target.file = $$BASE_DIR/HotWheels-integration-tests.pro
    integration_tests_target.depends = app_target
}
