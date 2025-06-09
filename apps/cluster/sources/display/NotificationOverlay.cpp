#include "NotificationOverlay.hpp"

NotificationOverlay::NotificationOverlay(QWidget* parent)
	: QWidget(parent)
{
	setAttribute(Qt::WA_TransparentForMouseEvents);
	setAttribute(Qt::WA_NoSystemBackground);
	setAttribute(Qt::WA_TranslucentBackground);
	setWindowFlags(Qt::FramelessWindowHint | Qt::ToolTip);
	resize(parent->size());
	opacityEffect = new QGraphicsOpacityEffect(this);
	setGraphicsEffect(opacityEffect);

	fadeAnimation = new QPropertyAnimation(opacityEffect, "opacity");
	fadeAnimation->setDuration(500);  // 500ms fade duration
}

void NotificationOverlay::showNotification(const QString& text, NotificationLevel notificationLevel, int durationMs)
{
	message = text;
	level = notificationLevel;
	show();
	update();

	// Fade in
	fadeAnimation->stop();
	fadeAnimation->setStartValue(0.0);
	fadeAnimation->setEndValue(1.0);
	fadeAnimation->start();

	// Schedule fade out after duration
	QTimer::singleShot(durationMs, this, [this]() {
		fadeAnimation->stop();
		fadeAnimation->setStartValue(1.0);
		fadeAnimation->setEndValue(0.0);
		fadeAnimation->start();

		// Fully hide after fade out completes
		connect(fadeAnimation, &QPropertyAnimation::finished, this, [this]() {
			if (opacityEffect->opacity() == 0.0)
				hide();
		});
	});
}

void NotificationOverlay::paintEvent(QPaintEvent*)
{
	QPainter painter(this);
	painter.setRenderHint(QPainter::Antialiasing);

	int boxWidth = width() * 0.3;
	int boxHeight = 80;
	int boxX = (width() - boxWidth) / 2;
	int boxY = 50;

	QRect rect(boxX, boxY, boxWidth, boxHeight);

	// Background only (no shadow to avoid nested effect)
	QColor backgroundColor(1, 32, 44, 255);
	painter.setBrush(backgroundColor);
	painter.setPen(Qt::NoPen);
	painter.drawRoundedRect(rect, 20, 20);

	// Icon
	QString iconPath;

	switch (level) {
		case NotificationLevel::Info:
			iconPath = ":/images/info.png";
			break;
		case NotificationLevel::Warning:
			iconPath = ":/images/warning.png";
			break;
	}

	QPixmap icon(iconPath);
	int iconSize = 40;
	int iconMargin = 20;
	int iconX = rect.left() + iconMargin;
	int iconY = rect.top() + (rect.height() - iconSize) / 2;
	painter.drawPixmap(iconX, iconY, iconSize, iconSize, icon);

	// Text
	QRect textRect = rect.adjusted(iconMargin + iconSize + 10, 0, -20, 0);
	painter.setPen(Qt::white);
	QFont font = painter.font();
	font.setPointSize(14);
	painter.setFont(font);
	painter.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, message);
}
