#include "tracker.hpp"
#include <opencv2/tracking.hpp>

Tracker::Tracker() : isInitialized_(false) {
    tracker_ = cv::TrackerCSRT::create();
}

void Tracker::init(const cv::Mat& frame, const cv::Rect& roi) {
    tracker_->init(frame, roi);
    isInitialized_ = true;
}

cv::Rect Tracker::update(const cv::Mat& frame) {
    if (!isInitialized_) {
        return cv::Rect();
    }
    
    cv::Rect bbox;
    if (tracker_->update(frame, bbox)) {
        return bbox;
    }
    
    isInitialized_ = false;
    return cv::Rect();
}

void Tracker::reset() {
    isInitialized_ = false;
    tracker_ = cv::TrackerCSRT::create();
} 