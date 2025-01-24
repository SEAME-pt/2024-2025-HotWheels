QT       += core gui widgets
CONFIG   += c++17

# Base directory (absolute path to the root of the project)
BASE_DIR = $$PWD

# Subprojects
TEMPLATE = subdirs
SUBDIRS += app_target tests_target

# App Subproject
app_target.file = $$BASE_DIR/HotWheels-app.pro

# Tests Subproject
tests_target.file = $$BASE_DIR/HotWheels-tests.pro
tests_target.depends = app_target
