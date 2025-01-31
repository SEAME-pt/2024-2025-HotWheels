#include "CarManager.hpp"
#include "enums.hpp"
#include <QApplication>
#include <QDebug>
#include <csignal>

// Signal handler for SIGINT (CTRL+C)
void handleSigint(int) {
  // qDebug() << "[Main] SIGINT received. Quitting application.";
  QCoreApplication::quit();
}

int main(int argc, char *argv[]) {
  qDebug() << "[Main] HotWheels Cluster starting...";

  std::signal(SIGINT, handleSigint);

  // Register enums
  qRegisterMetaType<ComponentStatus>("ComponentStatus");
  qRegisterMetaType<DrivingMode>("DrivingMode");
  qRegisterMetaType<ClusterTheme>("ClusterTheme");
  qRegisterMetaType<ClusterMetrics>("ClusterMetrics");
  qRegisterMetaType<CarDirection>("CarDirection");

  QApplication a(argc, argv);

  CarManager w;

  w.showFullScreen();

  int exitCode = a.exec();
  qDebug() << "[Main] HotWheels Cluster shutting down...";
  return exitCode;
}
