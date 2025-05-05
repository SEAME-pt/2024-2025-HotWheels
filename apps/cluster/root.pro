QT       += core gui widgets network
CONFIG   += c++17

# Base directory (absolute path to the root of the project)
BASE_DIR = $$PWD

# Subprojects
TEMPLATE = subdirs
SUBDIRS += \
	app_target \
	unit_tests_target \
	integration_tests_target

# App Subproject
app_target.file = $$BASE_DIR/HotWheels-app.pro

# Unit Tests Subproject
unit_tests_target.file = $$BASE_DIR/HotWheels-unit-tests.pro
unit_tests_target.depends = app_target

# Integration Tests Subproject
integration_tests_target.file = $$BASE_DIR/HotWheels-integration-tests.pro
integration_tests_target.depends = app_target
