#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <QApplication>
#include "NotificationManager.hpp"

using ::testing::_;
using ::testing::StrEq;
using ::testing::Eq;
using ::testing::Gt;

/*!
 * @class MockNotificationOverlay
 * @brief A mock class for NotificationOverlay.
 *
 * @details Allows interception of `showNotification` calls for test validation.
 */
class MockNotificationOverlay : public NotificationOverlay {
	Q_OBJECT
public:
	MockNotificationOverlay(QWidget* parent = nullptr) : NotificationOverlay(parent) {}

	MOCK_METHOD(void, showNotification, (const QString& text, NotificationLevel level, int durationMs), (override));
};

/*!
 * @class NotificationManagerTest
 * @brief Test fixture for NotificationManager tests.
 */
class NotificationManagerTest : public ::testing::Test {
protected:
	QApplication* app = nullptr;
	MockNotificationOverlay* mockOverlay = nullptr;
	NotificationManager* manager = nullptr;

	void SetUp() override {
		int argc = 0;
		app = new QApplication(argc, nullptr);  // Required for Qt widgets
		mockOverlay = new MockNotificationOverlay();
		manager = NotificationManager::instance();
		manager->initialize(mockOverlay);
	}

	void TearDown() override {
		delete mockOverlay;
		delete app;
	}
};

/*!
 * @test Tests that the NotificationManager singleton returns a non-null instance.
 * @brief Ensures that `NotificationManager::instance()` returns the same instance.
 */
TEST_F(NotificationManagerTest, SingletonReturnsValidInstance) {
	auto* instance1 = NotificationManager::instance();
	auto* instance2 = NotificationManager::instance();
	EXPECT_NE(instance1, nullptr);
	EXPECT_EQ(instance1, instance2);
}

/*!
 * @test Tests that a notification is shown when an overlay is initialized.
 * @brief Ensures that `showNotification()` is called with correct parameters.
 */
TEST_F(NotificationManagerTest, EnqueueNotification_CallsOverlay) {
	EXPECT_CALL(*mockOverlay, showNotification(QString("Test message"), NotificationLevel::Info, Eq(500)))
		.Times(1);

	manager->enqueueNotification("Test message", NotificationLevel::Info, 500);
}

/*!
 * @test Tests that no notification is shown if no overlay is set.
 * @brief Ensures that `showNotification()` is not called when the overlay is null.
 */
TEST(NotificationManagerStandaloneTest, EnqueueNotification_WithoutInitialization_DoesNothing) {
	NotificationManager* mgr = NotificationManager::instance();
	mgr->initialize(nullptr);  // Remove overlay

	EXPECT_NO_FATAL_FAILURE({
		mgr->enqueueNotification(QString("Warning!"), NotificationLevel::Warning, 300);
	});
}

