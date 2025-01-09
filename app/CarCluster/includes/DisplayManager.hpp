#ifndef DISPLAYMANAGER_HPP
#define DISPLAYMANAGER_HPP

#include <QLabel>
#include <QObject>
#include <QQuickWidget>
#include <QTimer>
#include <QWidget>

class MeterController;

class DisplayManager : public QObject
{
    Q_OBJECT

public:
    explicit DisplayManager(QWidget *parent,
                            QWidget *speedParentWidget,
                            MeterController *speedController,
                            QWidget *rpmParentWidget,
                            MeterController *rpmController,
                            QLabel *systemInfoLabel);
    ~DisplayManager();

private:
    MeterController *m_speedController;
    QQuickWidget *m_speedWidget;
    QWidget *m_speedParentWidget;

    MeterController *m_rpmController;
    QQuickWidget *m_rpmWidget;
    QWidget *m_rpmParentWidget;

    QLabel *m_systemInfoLabel;
    QTimer *m_timeUpdateTimer;

    void setupWidgets();
    void setupTimeLabel();
    void updateStatusDisplay();
};

#endif // DISPLAYMANAGER_HPP
