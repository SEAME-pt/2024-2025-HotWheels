#include "MockNotificationOverlay.hpp"
#include <QDebug>

MockNotificationOverlay::MockNotificationOverlay(QWidget* parent)
	: NotificationOverlay(parent) {
		qDebug() << "MockNotificationOverlay created";
}
