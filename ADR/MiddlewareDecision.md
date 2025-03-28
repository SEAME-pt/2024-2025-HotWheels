# Middleware choice

## ZeroC vs ZeroMQ

First, ZeroC was implemented and working but it wasn't efficient, as we had a thread checking the value every 100 ms, which could overload the CPU with so many requests.

Now, ZeroC is implemented using a publish/subscriber method, which means that the publisher sends a message to all the subscribers with the updated value.

ZeroMQ seems a better option as it has almost no latency when changing the values compared to ZeroC, where it was more difficult to implement the pub/sub system,
with the implementation described above.
