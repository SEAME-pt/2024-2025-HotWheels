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

  /* try {
    ControlsManager *m_controlsManager;
    m_controlsManager = new ControlsManager(argc, argv);
    return a.exec();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "Service has stopped." << std::endl; */

  try {
    // Create ControlsManager on the STACK instead of HEAP
    ControlsManager m_controlsManager(argc, argv);

    // Run the application event loop
    int result = a.exec();

    qDebug() << "Service has stopped.";
    return result;
  } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
  }
}
