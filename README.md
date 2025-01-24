# HotWheels - Instrument Cluster ⏲ 🎮
Welcome to the Instrument Cluster repository of the HotWheels team! Here you'll find all the work done during the SEA:ME Instrument Cluster project.  

Main repository: https://github.com/SEAME-pt/2024-2025-HotWheels

## Project Description
The HotWheels Instrument Cluster project aims to develop a sophisticated instrument cluster for a model car. The project focuses on integrating various components such as sensors, motors, and displays to create a functional and interactive system. The primary goals of the project are to enhance the driving experience, provide real-time data to the driver, and ensure seamless communication between different components.

## Communication Architecture
The communication architecture of the project is based on the CAN-BUS protocol. CAN-BUS (Controller Area Network) is a robust vehicle bus standard designed to allow microcontrollers and devices to communicate with each other without a host computer. In this project, CAN-BUS is used for communication between the Nvidea board and the arduino.  
  
This communication system and its components can be visualized as follows:

![image](https://github.com/user-attachments/assets/8a0310a5-d845-49f4-b54a-a813153147cf)

## Software Architecture
The software architecture of the project is designed to leverage the capabilities of the Nvidea board. The decision to use this board was based on its powerful GPU and AI tools, which are essential for advanced computer vision and sensor fusion tasks, providing the necessary processing power to run the main control operations for the motors and other components.

Our software architecture is simple, easy to understand and well organized allowing all team members to work on the code simultaneously and the expansion of the app for the future modules. 

![software-arch](https://github.com/user-attachments/assets/f113d084-b909-42c2-b1a0-845ba5ca4f2a)

## Adaptation to Read CAN Messages Using SPI Pins
Unfortunately this version of the Jetson Nano doesn't provide the CAN interface necessary to read CAN messages directly. Being so we were forced to improvise and adapt the QT app to read CAN messages using SPI (Serial Peripheral Interface) pins.  
This adaptation allows the system to interface with the CAN-BUS protocol through the pins on the Nvidea board guaranteeing proper communication between the components.

## Cross-Compilation Method
To ensure compatibility with the Nvidea board, the project employs a cross-compilation method from x86 to aarm64. This method involves compiling the software on an x86 machine and then deploying it on the aarm64 architecture of the board ensuring that the software runs efficiently on the target hardware.

## Results
Below is a small video of our final results.  

https://github.com/user-attachments/assets/36f429f6-f533-4ab9-9822-74d56cfc7189
