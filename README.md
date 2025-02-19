# HotWheels 🏁 🏎️
#### 🤝 Authorship - Collaborative Work
> Project done in collaboration with [Ricardo Melo](https://github.com/reomelo), [Félix Bihan](https://github.com/fle-bihh), [Tiago Pereira](https://github.com/t-pereira06) and [Michel Batista](https://github.com/MicchelFAB)

## Introduction
Welcome to the main repository of the HotWheels team! Here you'll find all the details regarding the autonomous driving JetRacer project developed by Team01 (HotWheels) during the [SEA:ME](https://seame.space/) program in Portugal (2024-2025).  

### What is SEA:ME?
SEA:ME is a cutting-edge, tuition-free advanced studies program in which students develop expertise in Code-driven Mobility, Autonomous Mobility, and Future Mobility Ecosystems. Over a 6-12 month period, students engage in expert-designed, practical coursework emphasizing hands-on, peer-to-peer learning in a lab environment. This immersive approach prepares them for work in the dynamic field of mobility IT.  

## Project Description
The main goal of the SEA:ME program is to develop a fully autonomous driving car. The car's communication system is based on the CAN(Controller Area Network) protocol which is a robust vehicle bus standard designed to allow microcontrollers and devices to communicate with each other without a host computer.  
  
In this project, we have a Nvidea board that is responsible for all the decision making and image processing tasks taking advantage of it's powerful GPU and AI tools. This main board gets the essential data from a microcontroller called arduino that is reponsible for communicating with the sensors (speed and ultrasound) and send the data via the CAN-BUS protocol.  
  
To achieve all this the project is divided into main modules: Cluster Instrument, Lane Detection, Object Detection and Lane Segmentation. These modules have their own readme file that you can check below.

## Modules
> ### [Instrument Cluster]() ⏲ 🎮  
> ### [Lane Detection]() 🛣️ 📷  
> ### [Object Detection]()  
> ### [Road Segmentation]()  