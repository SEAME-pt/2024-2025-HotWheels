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
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CarCluster
{
public:
    QWidget *centralWidget;
    QVBoxLayout *mainVerticalLayout;
    QSpacerItem *topSpacer;
    QHBoxLayout *mainHorizontalLayout;
    QSpacerItem *leftSpacer;
    QWidget *rpmMeterWidget;
    QLabel *systemInfoLabel;
    QWidget *speedMeterWidget;
    QSpacerItem *rightSpacer;
    QSpacerItem *bottomSpacer;

    void setupUi(QMainWindow *CarCluster)
    {
        if (CarCluster->objectName().isEmpty())
            CarCluster->setObjectName(QString::fromUtf8("CarCluster"));
        CarCluster->resize(442, 222);
        centralWidget = new QWidget(CarCluster);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        centralWidget->setAutoFillBackground(false);
        centralWidget->setStyleSheet(QString::fromUtf8("background-color: #000080;"));
        mainVerticalLayout = new QVBoxLayout(centralWidget);
        mainVerticalLayout->setObjectName(QString::fromUtf8("mainVerticalLayout"));
        topSpacer = new QSpacerItem(0, 4, QSizePolicy::Minimum, QSizePolicy::Expanding);

        mainVerticalLayout->addItem(topSpacer);

        mainHorizontalLayout = new QHBoxLayout();
        mainHorizontalLayout->setSpacing(6);
        mainHorizontalLayout->setObjectName(QString::fromUtf8("mainHorizontalLayout"));
        mainHorizontalLayout->setSizeConstraint(QLayout::SetMaximumSize);
        leftSpacer = new QSpacerItem(4, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        mainHorizontalLayout->addItem(leftSpacer);

        rpmMeterWidget = new QWidget(centralWidget);
        rpmMeterWidget->setObjectName(QString::fromUtf8("rpmMeterWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(rpmMeterWidget->sizePolicy().hasHeightForWidth());
        rpmMeterWidget->setSizePolicy(sizePolicy);
        rpmMeterWidget->setMinimumSize(QSize(150, 150));
        rpmMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));

        mainHorizontalLayout->addWidget(rpmMeterWidget);

        systemInfoLabel = new QLabel(centralWidget);
        systemInfoLabel->setObjectName(QString::fromUtf8("systemInfoLabel"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(systemInfoLabel->sizePolicy().hasHeightForWidth());
        systemInfoLabel->setSizePolicy(sizePolicy1);
        QFont font;
        font.setPointSize(10);
        systemInfoLabel->setFont(font);
        systemInfoLabel->setStyleSheet(QString::fromUtf8("color: rgb(246, 245, 244);"));
        systemInfoLabel->setAlignment(Qt::AlignCenter);

        mainHorizontalLayout->addWidget(systemInfoLabel);

        speedMeterWidget = new QWidget(centralWidget);
        speedMeterWidget->setObjectName(QString::fromUtf8("speedMeterWidget"));
        sizePolicy.setHeightForWidth(speedMeterWidget->sizePolicy().hasHeightForWidth());
        speedMeterWidget->setSizePolicy(sizePolicy);
        speedMeterWidget->setMinimumSize(QSize(150, 150));
        speedMeterWidget->setStyleSheet(QString::fromUtf8("background-color: transparent;"));

        mainHorizontalLayout->addWidget(speedMeterWidget);

        rightSpacer = new QSpacerItem(4, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        mainHorizontalLayout->addItem(rightSpacer);


        mainVerticalLayout->addLayout(mainHorizontalLayout);

        bottomSpacer = new QSpacerItem(0, 4, QSizePolicy::Minimum, QSizePolicy::Expanding);

        mainVerticalLayout->addItem(bottomSpacer);

        CarCluster->setCentralWidget(centralWidget);

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
