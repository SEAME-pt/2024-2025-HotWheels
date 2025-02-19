/*!
 * @file main.cpp
 * @brief Main function for the car controls service.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the main function for the car controls service.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <QCoreApplication>
#include <csignal>
#include <iostream>

volatile bool keepRunning = true;

/*!
 * @brief SIGINT signal handler.
 * @details This function will be called when the SIGINT signal is received.
 * The function will quit the QCoreApplication.
 */
void handleSigint(int) {
	qDebug() << "SIGINT received. Quitting application...";
	QCoreApplication::quit();
}

/*!
 * @brief Entry point for the car controls service.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return An integer indicating the exit status of the application.
 * @details Initializes the QCoreApplication and sets up signal handling for
 * SIGINT. Instantiates the ControlsManager and starts the event loop. If an
 * exception is thrown during execution, it is caught and logged, returning a
 * non-zero exit status. The application runs until quit is invoked.
 */

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
