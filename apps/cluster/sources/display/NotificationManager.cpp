#include "NotificationManager.hpp"

NotificationManager* NotificationManager::instance()
{
	static NotificationManager instanceObj;
	return &instanceObj;
}

NotificationManager::NotificationManager(QObject* parent)
	: QObject(parent)
{
}

void NotificationManager::initialize(NotificationOverlay* overlay)
{
	m_overlay = overlay;
}

void NotificationManager::enqueueNotification(const QString& text, NotificationLevel level, int durationMs)
{
	if (!m_overlay) return;

	m_queue.enqueue(std::make_tuple(text, level, durationMs));
	if (!m_busy) {
		showNext();
	}
}

void NotificationManager::showNext()
{
	if (m_queue.isEmpty()) {
		m_busy = false;
		return;
	}

	m_busy = true;
	auto [text, level, duration] = m_queue.dequeue();

	m_overlay->showNotification(text, level, duration); // Show for 2 seconds

	QTimer::singleShot(duration, this, [this]() {
		showNext();
	});
}
