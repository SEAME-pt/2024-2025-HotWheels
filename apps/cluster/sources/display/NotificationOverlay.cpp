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

	fadeOutTimer = new QTimer(this);
	fadeOutTimer->setSingleShot(true);
}

NotificationOverlay::~NotificationOverlay() = default;

void NotificationOverlay::showNotification(const QString& text, NotificationLevel notificationLevel, int durationMs)
{
	message = text;
	level = notificationLevel;

	fadeAnimation->stop();
	disconnect(fadeAnimation, nullptr, nullptr, nullptr);
	QMetaObject::invokeMethod(this, [this, durationMs]() {
		fadeOutTimer->stop();
		fadeOutTimer->start(durationMs);
	}, Qt::QueuedConnection);

	show();
	update();

	// Fade in
	fadeAnimation->setStartValue(0.0);
	fadeAnimation->setEndValue(1.0);
	fadeAnimation->start();

	connect(fadeOutTimer, &QTimer::timeout, this, [this]() {
		fadeAnimation->stop();
		disconnect(fadeAnimation, nullptr, nullptr, nullptr);

		fadeAnimation->setStartValue(1.0);
		fadeAnimation->setEndValue(0.0);
		fadeAnimation->start();

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

	// Setup font
	QFont font = painter.font();
	font.setPointSize(12);
	painter.setFont(font);

	// Measure text width
	QFontMetrics metrics(font);
	int textWidth = metrics.horizontalAdvance(message);

	// Icon parameters
	int iconSize = 30;
	int iconMargin = 20;

	// Compute box width based on text and icon
	int boxWidth = iconMargin + iconSize + 10 + textWidth + 20;  // iconMargin + icon + spacing + text + right margin
	int maxWidth = width() * 0.8;
	boxWidth = std::min(boxWidth, maxWidth);

	int boxHeight = 80;
	int boxX = (width() - boxWidth) / 2;
	int boxY = 50;

	QRect rect(boxX, boxY, boxWidth, boxHeight);

	// Background
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
	int iconX = rect.left() + iconMargin;
	int iconY = rect.top() + (rect.height() - iconSize) / 2;
	painter.drawPixmap(iconX, iconY, iconSize, iconSize, icon);

	// Text
	QRect textRect = rect.adjusted(iconMargin + iconSize + 10, 0, -20, 0);
	painter.setPen(Qt::white);
	painter.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, message);
}
