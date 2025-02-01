/**
 * @file main.cpp
 * @brief Main file of the HotWheels Cluster application.
 * @version 0.1
 * @date 2025-01-31
 * @details This file contains the main function of the HotWheels Cluster
 * application.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 * @warning Ensure that the CarManager class is properly implemented.
 * @see CarManager.hpp, enums.hpp
 * @copyright Copyright (c) 2025
 */

#include "CarManager.hpp"
#include "enums.hpp"
#include <QApplication>
#include <QDebug>
#include <csignal>

/**
 * @brief Signal handler for SIGINT.
 * @param signal The signal number.
 * @details This function is called when a SIGINT signal is received.
 */
void handleSigint(int) {
	// qDebug() << "[Main] SIGINT received. Quitting application.";
	QCoreApplication::quit();
}

/**
 * @brief Main function of the HotWheels Cluster application.
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return int The exit code of the application.
 * @details This function initializes the application and starts the CarManager.
 */
int main(int argc, char *argv[]) {
	qDebug() << "[Main] HotWheels Cluster starting...";

	if (std::signal(SIGINT, handleSigint) == SIG_ERR) {
		qDebug() << "[Main] Error setting up signal handler.";
		return 1;
	}

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
