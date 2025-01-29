#include <QCoreApplication>
#include "ControlsManager.hpp"
#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>

volatile bool keepRunning = true;

// Signal handler for SIGINT (CTRL+C)
void handleSigint(int)
{
    qDebug() << "[Main] SIGINT received. Quitting application.";
    keepRunning = false;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::signal(SIGINT, handleSigint);

    try {
        ControlsManager *m_controlsManager;

        m_controlsManager = new ControlsManager();
        while (keepRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Service has stopped." << std::endl;

    return a.exec();
}
