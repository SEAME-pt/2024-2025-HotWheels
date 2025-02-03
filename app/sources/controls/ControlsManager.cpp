/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @details This file contains the implementation of the ControlsManager class,
 *          which is used to manage the controls of the vehicle.
 * @version 0.1
 * @date 2025-01-31
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @note This class is used to manage the controls of the vehicle.
 *
 * @warning Ensure that the EngineController and JoysticksController classes are
 * properly implemented.
 *
 * @see ControlsManager.hpp for the class definition.
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <QDebug>

#define SHM_NAME "/joystick_enable"

/*!
 * @brief Construct a new ControlsManager object.
 * @param parent The parent QObject.
 * @details This constructor initializes the ControlsManager object.
 */
ControlsManager::ControlsManager(QObject *parent)
    : QObject(parent) {

    // Create shared memory object
    this->shm_fd = shm_open("/joystick_enable", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory\n";
    }

    // Set size of shared memory
    if (ftruncate(this->shm_fd, sizeof(bool)) == -1) {
        std::cerr << "Failed to set size\n";
    }

    // Map shared memory
    this->ptr = mmap(0, sizeof(bool), PROT_READ | PROT_WRITE, MAP_SHARED, this->shm_fd, 0);
    if (this->ptr == MAP_FAILED) {
        std::cerr << "Failed to map memory\n";
    }

    // Write to shared memory (set bool value)
    *(static_cast<bool*>(this->ptr)) = true;
}

/*!
 * @brief Destroy the ControlsManager object.
 * @details This destructor stops the joystick controller and waits for the
 * thread to finish.
 */
ControlsManager::~ControlsManager() {
  // Cleanup of shared memory
  if (this->ptr)
    munmap(ptr, sizeof(bool));
  if (this->shm_fd != -1)
  {
    close(this->shm_fd);
    shm_unlink(SHM_NAME);
  }
}

/*!
 * @brief Update the driving mode of the vehicle.
 * @param newMode The new driving mode of the vehicle.
 * @details This slot is called when the driving mode of the vehicle is changed.
 *          It updates the current driving mode by calling the setMode() method.
 */
void ControlsManager::drivingModeUpdated(DrivingMode newMode) {
  int shm_fd = shm_open("/joystick_enable", O_RDWR, 0666);
  if (shm_fd == -1) {
      std::cerr << "Failed to open shared memory\n";
  }

  // Map shared memory
  void* ptr = mmap(0, sizeof(bool), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
      std::cerr << "Failed to map memory\n";
  }

  // Modify the shared memory
  if (newMode == DrivingMode::Automatic)
    *flag = false;
  else
    *flag = true;

  // Cleanup
  munmap(ptr, sizeof(bool));
  close(shm_fd);
}
