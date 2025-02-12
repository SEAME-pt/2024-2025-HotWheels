/*!
 * @file FileController.hpp
 * @brief Definition of the FileController namespace. 
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the definition of the FileController namespace, which
 * is responsible for handling file operations.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef FILECONTROLLER_HPP
#define FILECONTROLLER_HPP

#include <QFile>
#include <QString>

/*!
 * @brief Namespace containing file handling functions.
 * @namespace FileController
 */
namespace FileController {
bool open(QFile &file, QIODevice::OpenMode mode);
QString read(QFile &file);
bool write(QFile &file, const QString &data);
bool exists(const QString &path);
} // namespace FileController

#endif // FILECONTROLLER_HPP
