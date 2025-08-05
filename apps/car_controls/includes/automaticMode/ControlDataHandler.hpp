#ifndef CONTROL_DATA_HANDLER_HPP
#define CONTROL_DATA_HANDLER_HPP

#include <string>
#include <vector>
#include "../../ZeroMQ/Subscriber.hpp"
#include "CommonTypes.hpp"

class ControlDataHandler {
public:
    // === Constructors and Destructor ===
    ControlDataHandler();
    ~ControlDataHandler();

    // === Connection Management ===
    void initializeSubscribers();
    void cleanupSubscribers();

    // === Data Retrieval Methods ===
    CenterlineResult getPolyfittingResult();
    bool getShouldSlowDown() const;

private:
    // === Subscriber Settings ===
    const std::string POLYFITTING_TOPIC = "polyfitting_result";
    const std::string POLYFITTING_PORT = "tcp://localhost:5569";

    const std::string OBJECT_TOPIC = "notification";
    const std::string OBJECT_PORT = "tcp://localhost:5557";

    const std::string SLOW_DOWN_OBJECTS[4] = {"danger", "ceding", "crosswalk", "yellow"};

    // === Subscribers ===
    Subscriber *m_polyfittingSubscriber;
    Subscriber *m_objectDetectionSubscriber;

    // === JSON Parsing Methods ===
    CenterlineResult extractJsonData(std::string data);
    std::vector<Point2D> parsePointArray(const std::string& json_data, const std::string& field_name);
    std::vector<LaneCurve> parseLaneArray(const std::string& json_data);
};

#endif