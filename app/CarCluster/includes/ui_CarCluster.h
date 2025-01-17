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
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CarCluster
{
public:
    QWidget *clusterWindowWidget;
    QWidget *speedMeterWidget;
    QWidget *rpmMeterWidget;
    QLabel *systemInfoLabel;

    void setupUi(QMainWindow *CarCluster)
    {
        if (CarCluster->objectName().isEmpty())
            CarCluster->setObjectName(QString::fromUtf8("CarCluster"));
        CarCluster->resize(1024, 600);
        CarCluster->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        clusterWindowWidget = new QWidget(CarCluster);
        clusterWindowWidget->setObjectName(QString::fromUtf8("clusterWindowWidget"));
        clusterWindowWidget->setAutoFillBackground(false);
        clusterWindowWidget->setStyleSheet(QString::fromUtf8("background-color: #000957;"));
        speedMeterWidget = new QWidget(clusterWindowWidget);
        speedMeterWidget->setObjectName(QString::fromUtf8("speedMeterWidget"));
        speedMeterWidget->setGeometry(QRect(644, 230, 290, 300));
        speedMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        rpmMeterWidget = new QWidget(clusterWindowWidget);
        rpmMeterWidget->setObjectName(QString::fromUtf8("rpmMeterWidget"));
        rpmMeterWidget->setGeometry(QRect(80, 230, 300, 300));
        rpmMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));
        systemInfoLabel = new QLabel(clusterWindowWidget);
        systemInfoLabel->setObjectName(QString::fromUtf8("systemInfoLabel"));
        systemInfoLabel->setGeometry(QRect(312, 50, 400, 130));
        QFont font;
        font.setPointSize(14);
        systemInfoLabel->setFont(font);
        systemInfoLabel->setLayoutDirection(Qt::LeftToRight);
        systemInfoLabel->setAutoFillBackground(false);
        systemInfoLabel->setStyleSheet(QString::fromUtf8("color: rgb(246, 245, 244); background-color: transparent"));
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
