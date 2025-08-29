#ifndef CONTROL_DATA_HANDLER_HPP
#define CONTROL_DATA_HANDLER_HPP

#include <string>
#include <vector>
#include <QThread>
#include "../../ZeroMQ/Subscriber.hpp"
#include "../../ZeroMQ/CommonTypes.hpp"

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
    float getCarSpeed() const;

    double latestUltrasoundMeters() const {
        return m_latestUltraMeters.load(std::memory_order_relaxed);
    }

private:
    // === Subscriber Settings ===
    const std::string SLOW_DOWN_OBJECTS[4] = {"danger", "ceding", "crosswalk", "yellow"};

    // === Subscribers ===
    Subscriber *m_polyfittingSubscriber;
    Subscriber *m_objectDetectionSubscriber;
    Subscriber *m_carSpeedSubscriber;
    Subscriber *m_ultrasoundSubscriber;

    QThread *m_ultrasoundThread;
    std::atomic<bool>   m_ultraRunning { false };
    std::atomic<double> m_latestUltraMeters { std::numeric_limits<double>::quiet_NaN() };

    // === JSON Parsing Methods ===
    CenterlineResult extractJsonData(std::string data);
    std::vector<Point2D> parsePointArray(const std::string& json_data, const std::string& field_name);
    std::vector<LaneCurve> parseLaneArray(const std::string& json_data);
};

#endif
