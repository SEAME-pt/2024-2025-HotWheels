/********************************************************************************
** Form generated from reading UI file 'CarCluster.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CARCLUSTER_H
#define UI_CARCLUSTER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CarCluster
{
public:
    QWidget *clusterWindowWidget;
    QFrame *sliderFrame;
    QWidget *speedMeterWidget;
    QWidget *rpmMeterWidget;
    QLabel *systemInfoLabel;

    void setupUi(QMainWindow *CarCluster)
    {
        if (CarCluster->objectName().isEmpty())
            CarCluster->setObjectName(QString::fromUtf8("CarCluster"));
        CarCluster->resize(1080, 400);
        clusterWindowWidget = new QWidget(CarCluster);
        clusterWindowWidget->setObjectName(QString::fromUtf8("clusterWindowWidget"));
        clusterWindowWidget->setAutoFillBackground(false);
        clusterWindowWidget->setStyleSheet(QString::fromUtf8("background-color: #000080;"));
        sliderFrame = new QFrame(clusterWindowWidget);
        sliderFrame->setObjectName(QString::fromUtf8("sliderFrame"));
        sliderFrame->setGeometry(QRect(20, 339, 371, 41));
        sliderFrame->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        sliderFrame->setFrameShape(QFrame::StyledPanel);
        sliderFrame->setFrameShadow(QFrame::Raised);
        speedMeterWidget = new QWidget(clusterWindowWidget);
        speedMeterWidget->setObjectName(QString::fromUtf8("speedMeterWidget"));
        speedMeterWidget->setGeometry(QRect(760, 90, 300, 300));
        speedMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        rpmMeterWidget = new QWidget(clusterWindowWidget);
        rpmMeterWidget->setObjectName(QString::fromUtf8("rpmMeterWidget"));
        rpmMeterWidget->setGeometry(QRect(10, 80, 300, 300));
        rpmMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        systemInfoLabel = new QLabel(clusterWindowWidget);
        systemInfoLabel->setObjectName(QString::fromUtf8("systemInfoLabel"));
        systemInfoLabel->setGeometry(QRect(100, 20, 841, 51));
        QFont font;
        font.setPointSize(10);
        systemInfoLabel->setFont(font);
        systemInfoLabel->setLayoutDirection(Qt::LeftToRight);
        systemInfoLabel->setStyleSheet(QString::fromUtf8("color: rgb(246, 245, 244)"));
        systemInfoLabel->setAlignment(Qt::AlignCenter);
        CarCluster->setCentralWidget(clusterWindowWidget);

        retranslateUi(CarCluster);

        QMetaObject::connectSlotsByName(CarCluster);
    } // setupUi

    void retranslateUi(QMainWindow *CarCluster)
    {
        CarCluster->setWindowTitle(QCoreApplication::translate("CarCluster", "CarCluster", nullptr));
        systemInfoLabel->setText(QCoreApplication::translate("CarCluster", "Current Time", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CarCluster: public Ui_CarCluster {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CARCLUSTER_H
