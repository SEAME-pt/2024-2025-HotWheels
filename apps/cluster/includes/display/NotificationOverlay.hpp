#ifndef NOTIFICATIONOVERLAY_HPP
#define NOTIFICATIONOVERLAY_HPP

#pragma once

#include <QWidget>
#include <QTimer>
#include <QPainter>
#include <QGraphicsOpacityEffect>
#include <QPropertyAnimation>

enum class NotificationLevel {
	Info,
	Warning
};

class NotificationOverlay : public QWidget {
	Q_OBJECT

public:
	explicit NotificationOverlay(QWidget* parent = nullptr);
	virtual ~NotificationOverlay();
	virtual void showNotification(const QString& text, NotificationLevel notificationLevel, int durationMs = 2000);

protected:
	void paintEvent(QPaintEvent* event) override;

private:
	NotificationLevel level = NotificationLevel::Info;
	QString message;
	QGraphicsOpacityEffect* opacityEffect;
	QPropertyAnimation* fadeAnimation;
	QTimer* fadeOutTimer;
};

#endif // NOTIFICATIONOVERLAY_HPP
