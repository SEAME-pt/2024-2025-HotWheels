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

	/* const int MAX_QUEUE_SIZE = 5;
	if (m_queue.size() >= MAX_QUEUE_SIZE) {
		m_queue.dequeue();  // Drop oldest to make room
	}

	// Prevent duplicate text
	bool alreadyQueued = std::any_of(m_queue.begin(), m_queue.end(), [&](const auto& t) {
		return std::get<0>(t) == text;
	});

	if (!alreadyQueued) {
		m_queue.enqueue(std::make_tuple(text, level, durationMs));
	}

	m_queue.enqueue(std::make_tuple(text, level, durationMs));
	if (!m_busy) {
		showNext();
	} */

	m_overlay->showNotification(text, level, durationMs);
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
