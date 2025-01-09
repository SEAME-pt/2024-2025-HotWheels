#include "CarCluster.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CarCluster w;
    w.showFullScreen();
    return a.exec();
}
