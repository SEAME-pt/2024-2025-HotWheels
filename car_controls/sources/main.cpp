#include "ControlsManager.hpp"
#include <QCoreApplication>
#include <csignal>
#include <iostream>

volatile bool keepRunning = true;

// Signal handler for SIGINT (CTRL+C)
void handleSigint(int) {
  qDebug() << "SIGINT received. Quitting application...";
  QCoreApplication::quit();
}

int main(int argc, char *argv[]) {
  QCoreApplication a(argc, argv);
  std::signal(SIGINT, handleSigint);

  try {
    ControlsManager *m_controlsManager;
    m_controlsManager = new ControlsManager(argc, argv);

    return a.exec();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "Service has stopped." << std::endl;
}
