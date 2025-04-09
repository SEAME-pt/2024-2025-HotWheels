# Middleware choice

## ZeroC vs ZeroMQ

First, ZeroC was implemented and functional, but it proved inefficient due to its design, which required a thread to poll for updates every 100 ms. This approach risked overloading the CPU with frequent requests, especially under high demand.

In contrast, ZeroMQ has been implemented using a publish/subscribe (pub/sub) model. This design allows the publisher to broadcast updated values to all subscribers in real time, eliminating the need for constant polling.

ZeroMQ emerges as the superior choice due to its minimal latency when propagating value changes. Additionally, implementing the pub/sub system with ZeroMQ was more straightforward compared to ZeroC, which required a more complex and less efficient setup.
