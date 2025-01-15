#include "CarCluster.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CarCluster w;
    // w.show();

    w.showFullScreen();
    return a.exec();
}
