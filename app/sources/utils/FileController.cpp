/*!
 * @file FileController.cpp
 * @brief 
 * @version 0.1
 * @date 2025-02-12
 * @details 
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "FileController.hpp"
#include <QDebug>
#include <QTextStream>

/*!
 * @namespace FileController
 * @brief Open a file.
 */
namespace FileController {

	/*!
	 * @brief Open a file.
	 *
	 * @details Opens a file with the specified mode.
	 *
	 * @param file The file to open.
	 * @param mode The mode to open the file with.
	 *
	 * @return True if the file was opened successfully, false otherwise.
	 */
	bool open(QFile &file, QIODevice::OpenMode mode)
	{
		return file.open(mode);
	}

	/*!
	 * @brief Reads a line from the file.
	 *
	 * @details This function reads a single line from the specified file using
	 * a QTextStream. It assumes the file is already open for reading.
	 *
	 * @param file The file to read from.
	 *
	 * @return The line read from the file as a QString.
	 */
	QString read(QFile &file)
	{
		QTextStream in(&file);
		return in.readLine();
	}

	/*!
	 * @brief Writes a line to the file.
	 *
	 * @details This function writes a single line to the specified file using
	 * a QTextStream. It assumes the file is already open for writing.
	 *
	 * @param file The file to write to.
	 * @param data The line to write to the file.
	 *
	 * @return True if the line was written successfully, false otherwise.
	 */
	bool write(QFile &file, const QString &data)
	{
		QTextStream out(&file);
		out << data << Qt::endl;
		return true;
	}

	/*!
	 * @brief Checks if a file exists.
	 *
	 * @details This function checks if the given file path exists.
	 *
	 * @param path The file path to check.
	 *
	 * @return True if the file exists, false otherwise.
	 */
	bool exists(const QString &path)
	{
		return QFile::exists(path);
	}

} // namespace FileController
