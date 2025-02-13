/*!
 * @file MockSystemCommandExecutor.hpp
 * @brief File containing the Mock class of the SystemCommandExecutor class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the SystemCommandExecutor class.
 * It uses Google Mock to create mock methods for the SystemCommandExecutor class.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKSYSTEMCOMMANDEXECUTOR_HPP
#define MOCKSYSTEMCOMMANDEXECUTOR_HPP

#include "ISystemCommandExecutor.hpp"
#include <gmock/gmock.h>

/*!
 * @class MockSystemCommandExecutor
 * @brief Class to emulate the behavior of the SystemCommandExecutor class.
 */
class MockSystemCommandExecutor : public ISystemCommandExecutor
{
public:
	/*! @brief Mocked method to execute a command. */
	MOCK_METHOD(QString, executeCommand, (const QString &command), (const, override));
	/*! @brief Mocked method to read a file. */
	MOCK_METHOD(QString, readFile, (const QString &filePath), (const, override));
};

#endif // MOCKSYSTEMCOMMANDEXECUTOR_HPP
