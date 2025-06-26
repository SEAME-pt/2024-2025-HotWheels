#include "MockNotificationOverlay.hpp"

MockNotificationOverlay::MockNotificationOverlay(QWidget* parent)
	: NotificationOverlay(parent) {
		qDebug() << "MockNotificationOverlay created";
}
