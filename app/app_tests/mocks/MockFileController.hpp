/*!
 * @file MockFileController.hpp
 * @brief File containing Mock classes to test the controller of the File
 * module.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains a mock class for the FileController module.
 * It uses Google Mock to create mock methods for the FileController module.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef MOCKFILECONTROLLER_HPP
#define MOCKFILECONTROLLER_HPP

#include <QFile>
#include <QString>
#include <gmock/gmock.h>

/*!
 * @class MockFileController
 * @brief Class to emulate the behavior of the File controller.
 */
class MockFileController
{
public:
	/** @brief Mocked method to open a file. */
	MOCK_METHOD(bool, open, (QFile &, QIODevice::OpenMode), ());
	/** @brief Mocked method to close a file. */
	MOCK_METHOD(QString, read, (QFile &), ());
	/** @brief Mocked method to write to a file. */
	MOCK_METHOD(bool, write, (QFile &, const QString &), ());
	/** @brief Mocked method to check if a file exists. */
	MOCK_METHOD(bool, exists, (const QString &), ());
};

#endif // MOCKFILECONTROLLER_HPP
