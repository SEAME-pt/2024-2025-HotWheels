#ifndef CARMANAGER_H
#define CARMANAGER_H

#include "CanBusManager.hpp"
#include "ControlsManager.hpp"
#include "DataManager.hpp"
#include "DisplayManager.hpp"
#include "MileageManager.hpp"
#include "SystemManager.hpp"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class CarManager;
}
QT_END_NAMESPACE

class CarManager : public QMainWindow {
  Q_OBJECT

public:
  /**
   * Constructor for the CarManager class. Initializes all necessary components
   * and sets up the UI.
   *
   * @param parent Optional parent widget (default is nullptr).
   */
  CarManager(QWidget *parent = nullptr);

  /**
   * Destructor for the CarManager class. Cleans up and deletes allocated
   * resources.
   */
  ~CarManager();

private:
  Ui::CarManager *ui; /**< Pointer to the user interface for the car manager. */
  DataManager *m_dataManager;     /**< Pointer to the DataManager instance. */
  CanBusManager *m_canBusManager; /**< Pointer to the CanBusManager instance. */
  ControlsManager
      *m_controlsManager; /**< Pointer to the ControlsManager instance. */
  DisplayManager
      *m_displayManager; /**< Pointer to the DisplayManager instance. */
  SystemManager *m_systemManager; /**< Pointer to the SystemManager instance. */
  MileageManager
      *m_mileageManager; /**< Pointer to the MileageManager instance. */

  /**
   * Initializes all the components of the CarManager.
   * This function is called during construction to set up various managers and
   * establish connections.
   */
  void initializeComponents();

  /**
   * Initializes the DataManager component. This method is a placeholder for
   * future logic.
   */
  void initializeDataManager();

  /**
   * Initializes the CanBusManager and connects relevant signals to slots.
   * Ensures that CAN bus data (speed, RPM) is properly handled.
   */
  void initializeCanBusManager();

  /**
   * Initializes the ControlsManager and connects its signals to the
   * DataManager. Ensures that direction and steering data are communicated to
   * the DataManager.
   */
  void initializeControlsManager();

  /**
   * Initializes the DisplayManager and connects DataManager signals to
   * DisplayManager slots. Updates UI elements with CAN data, engine data,
   * system info, etc.
   */
  void initializeDisplayManager();

  /**
   * Initializes the SystemManager and connects its signals to the DataManager.
   * This manager handles system-specific data like time, Wi-Fi status,
   * temperature, and battery percentage.
   */
  void initializeSystemManager();

  /**
   * Initializes the MileageManager and connects CAN bus data (speed) to it.
   * Tracks mileage data and updates it accordingly.
   */
  void initializeMileageManager();
};

#endif // CARMANAGER_H
