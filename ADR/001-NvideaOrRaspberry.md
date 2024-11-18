# Nvidea or RaspberryPi

## Context
The first issue regarding the Instrument Cluster project was the decision between Nvidea or RaspberryPi for the main controller of the car itself.  
  
The cars that we were provided used the Nvidea interface, but we were also provided with the necessary equipment to implement a Raspberry interface if we wanted to. Both interfaces have their pros/cons and are most suitable to do different tasks inside the project, being even possible to use them simultaneously.
  
## Decision
After some extensive research, the team concluded the best approach would be to use both components together, allowing intercommunication through the CAN-BUS protocol.  
  
The Nvidea's GPU and AI tools are perfect for the future modules regarding object and lane detection supporting advanced computer vision and sensor fusion tasks.  
  
However, this requires a lot of processing power and overloading the Nvidea with all this plus the main control tasks would be too much and could cause lag problems and false data exchange, leading to the possible failure of crutial operations.  
  
Beeing so, we decided that the Raspberry would be responsible for the main control operations and the Nvidea would take care of the processes regarding visual detection. All the data exchange would be done using communication protocols such as CAN-BUS, SPI, Bluetooth and others.  
  
  
In conclusion our final architecture would be something similar to this:  
[CAMERA] --> CAN-BUS --> [NVIDEA] --> SPI --> [RASPBERRY] <-- CAN-BUS <-- [SENSORS]  
[RASPBERRY] --> CAN-BUS --> [MOTORS]  
[RASPBERRY] --> Bluetooth --> [DISPLAY]  
  
## Consequences
If implemented correctly this structure would provide a much better organization of the project and efficient usage of all the components allowing them to be explored to their max potential.  
Besides that, it makes the project easier to scale if the addition of more components is necessary. 
