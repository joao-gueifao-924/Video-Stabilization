#pragma once

#include <opencv2/opencv.hpp>

class Tracker {
public:
    Tracker();
    
    // Initialize the tracker with the first frame and ROI
    void init(const cv::Mat& frame, const cv::Rect& roi);
    
    // Update the tracker with a new frame
    cv::Rect update(const cv::Mat& frame);
    
    // Reset the tracker
    void reset();

private:
    cv::Ptr<cv::Tracker> tracker_;
    bool isInitialized_;
}; 