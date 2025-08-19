/*!
 * @file test_int_CanBusManager.cpp
 * @brief Integration tests for the CanBusManager class.
 * @version 0.1
 * @date 2025-02-12
 * @author Michel Batista (@MicchelFAB)
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Ricardo Melo (@reomelo)
 * @author Tiago Pereira (@t-pereira06)
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <QCoreApplication>
#include <QDebug>
#include <QSignalSpy>
#include "CanBusManager.hpp"
#include "MCP2515Controller.hpp"
#include <gtest/gtest.h>

/*!
 * @brief Class to test the integration between the CanBusManager and the
 * MCP2515 controller.
 * @class CanBusManagerTest
 */

class FakeMCP2515Controller : public IMCP2515Controller {
	Q_OBJECT
public:
	FakeMCP2515Controller() = default;
	~FakeMCP2515Controller() override = default;

	// Implement pure virtuals with no-ops
	bool init() override { return true; }
	void processReading() override {}
	void stopReading() override {}
	bool isStopReadingFlagSet() const override { return false; }

	// Helpers so tests can simulate hardware signals
	void emitSpeed(float value) { emit speedUpdated(value); }
	void emitRpm(int value) { emit rpmUpdated(value); }
};

class CanBusManagerTest : public ::testing::Test
{
protected:
	static QCoreApplication *app;
	CanBusManager *canBusManager;
	IMCP2515Controller *controller;

	static void SetUpTestSuite()
	{
		int argc = 0;
		char *argv[] = {nullptr};
		app = new QCoreApplication(argc, argv);
	}

	static void TearDownTestSuite() { delete app; }

	void SetUp() override
	{
		controller = new FakeMCP2515Controller();
		ASSERT_NE(controller, nullptr);

		canBusManager = new CanBusManager(controller);
		ASSERT_NE(canBusManager, nullptr);
	}

	void TearDown() override
	{
		delete canBusManager;
		delete controller;
	}
};

/*! @brief Initialize static member */
QCoreApplication *CanBusManagerTest::app = nullptr;

/*!
 * @test 🚗 Forward Speed Data
 * @brief Ensures that the CanBusManager forwards speed data from the MCP2515
 * controller.
 * @details This test verifies that the CanBusManager forwards speed data from
 * the MCP2515 controller by emitting the speedUpdated signal.
 * @see CanBusManager::speedUpdated
 */
TEST_F(CanBusManagerTest, ForwardSpeedDataFromMCP2515)
{
	QSignalSpy spy(canBusManager, &CanBusManager::speedUpdated);

	emit controller->speedUpdated(88.8f);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_FLOAT_EQ(args.at(0).toFloat(), 88.8f);
}

/*!
 * @test 🔄 Forward RPM Data
 * @brief Ensures that the CanBusManager forwards RPM data from the MCP2515
 * controller.
 * @details This test verifies that the CanBusManager forwards RPM data from
 * the MCP2515 controller by emitting the rpmUpdated signal.
 * @see CanBusManager::rpmUpdated
 */
TEST_F(CanBusManagerTest, ForwardRpmDataFromMCP2515)
{
	QSignalSpy spy(canBusManager, &CanBusManager::rpmUpdated);

	emit controller->rpmUpdated(4500);
	QCoreApplication::processEvents();

	ASSERT_GT(spy.count(), 0) << "Signal was not emitted!";
	QList<QVariant> args = spy.takeFirst();
	EXPECT_EQ(args.at(0).toInt(), 4500);
}

/*!
 * @test 🚀 Initialization
 * @brief Ensures that the CanBusManager initializes successfully.
 * @details This test verifies that the CanBusManager initializes successfully
 * by calling the initialize() method.
 * @see CanBusManager::initialize
 */
TEST_F(CanBusManagerTest, InitializeCanBusManager)
{
	ASSERT_TRUE(canBusManager->initialize()) << "Initialization failed!";
	ASSERT_NE(canBusManager->getThread(), nullptr) << "Thread not created!";
	ASSERT_TRUE(canBusManager->getThread()->isRunning()) << "Thread did not start!";
}

/*!
 * @test 🧹 Manager Clean-Up Behavior
 * @brief Ensures that the CanBusManager cleans up properly.
 * @details This test verifies that the CanBusManager cleans up properly by
 * deleting the manager and checking if the thread was deleted.
 */
TEST_F(CanBusManagerTest, ManagerCleanUpBehavior)
{
	CanBusManager *tmpManager = new CanBusManager("/dev/spidev0.0");
	ASSERT_NE(tmpManager, nullptr);

	ASSERT_EQ(tmpManager->getThread(), nullptr) << "Thread created too soon!";

	tmpManager->initialize();

	ASSERT_NE(tmpManager->getThread(), nullptr) << "Thread not created!";
	ASSERT_TRUE(tmpManager->getThread()->isRunning()) << "Thread did not start!";

	delete tmpManager;

	ASSERT_EQ(tmpManager->getThread(), nullptr) << "Thread not deleted!";
}

#include "test_int_CanBusManager.moc"
