#ifndef NOTIFICATIONMANAGER_HPP
#define NOTIFICATIONMANAGER_HPP

#pragma once

#include <QObject>
#include <QQueue>
#include <QString>
#include "NotificationOverlay.hpp"

class NotificationManager : public QObject {
	Q_OBJECT

public:
	static NotificationManager* instance();

	void initialize(NotificationOverlay* overlay);
	void enqueueNotification(const QString& text, NotificationLevel level = NotificationLevel::Info, int durationMs = 3000);

private:
	explicit NotificationManager(QObject* parent = nullptr);

	NotificationOverlay* m_overlay = nullptr;
	QQueue<std::tuple<QString, NotificationLevel, int>> m_queue;
	bool m_busy = false;

	void showNext();
};

#endif // NOTIFICATIONMANAGER_HPP
