#ifndef BUTTONSCONTROLLER_HPP
#define BUTTONSCONTROLLER_HPP

#include <QObject>

class ButtonsController : public QObject
{
    Q_OBJECT

public:
    explicit ButtonsController(QObject *parent = nullptr);

signals:
    void drivingModeButtonClicked();
    void themeButtonClicked();
    void metricsButtonClicked();

public slots:
    void handleDrivingModeButton();
    void handleThemeButton();
    void handleMetricsButton();
};

#endif // BUTTONSCONTROLLER_HPP
