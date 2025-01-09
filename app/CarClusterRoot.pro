# CarClusterRoot.pro - Links all modules together

TEMPLATE = subdirs

SUBDIRS += CircularMeterModule \
           CarCluster

CarCluster.depends = CircularMeterModule

CircularMeterModule.path = $$OUT_PWD/CircularMeterModule
CarCluster.path = $$OUT_PWD/CarCluster

CONFIG += ordered
