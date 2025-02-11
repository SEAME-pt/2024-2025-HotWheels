#ifndef CLIENT_THREAD_HPP
#define CLIENT_THREAD_HPP

#include <Ice/Ice.h>
#include "Joystick.h"  // Generated by slice2cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <QObject>

class ClientThread : public QObject {
    Q_OBJECT

private:
    Ice::CommunicatorPtr communicator;
    Ice::ObjectPrx base;
    Data::CarDataPrx carData; // Proxy to communicate with the server
    std::thread clientThread;
    bool running = true;
    bool connected = false; // Flag to check if the client is connected
    std::mutex mtx;         // Mutex for synchronization
    std::condition_variable cv; // Condition variable for synchronization

public:
    explicit ClientThread(QObject *parent = nullptr);

    ~ClientThread();

    void runClient(int argc, char* argv[]);

    void setJoystickValue(bool value);
    bool getJoystickValue();

    setRunning(bool running);
};

#endif
