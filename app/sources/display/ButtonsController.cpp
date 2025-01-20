#include "ButtonsController.hpp"
#include <QDebug>

ButtonsController::ButtonsController(QObject *parent)
    : QObject(parent)
{}

void ButtonsController::handleDrivingModeButton()
{
    // qDebug() << "Driving mode button clicked.";
    emit drivingModeButtonClicked();
}

void ButtonsController::handleThemeButton()
{
    // qDebug() << "Theme button clicked.";
    emit themeButtonClicked();
}

void ButtonsController::handleMetricsButton()
{
    // qDebug() << "Metrics button clicked.";
    emit metricsButtonClicked();
}
