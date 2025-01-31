#include "ControlsManager.hpp"
#include <QCoreApplication>
#include <atomic>
#include <csignal>
#include <iostream>
#include <thread>

volatile bool keepRunning = true;

// Signal handler for SIGINT (CTRL+C)
void handleSigint(int)
{
    qDebug() << "SIGINT received. Quitting application...";
    QCoreApplication::quit();
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    std::signal(SIGINT, handleSigint);

    try {
        ControlsManager *m_controlsManager;
        m_controlsManager = new ControlsManager();
        return a.exec();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Service has stopped." << std::endl;
}
