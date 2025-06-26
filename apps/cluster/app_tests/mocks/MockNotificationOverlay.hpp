#pragma once

#include <gmock/gmock.h>
#include "NotificationOverlay.hpp"

class MockNotificationOverlay : public NotificationOverlay {
	Q_OBJECT

public:
	explicit MockNotificationOverlay(QWidget* parent = nullptr);
	~MockNotificationOverlay() override = default;

	MOCK_METHOD(void, showNotification, (const QString& text, NotificationLevel level, int durationMs), (override));
};
