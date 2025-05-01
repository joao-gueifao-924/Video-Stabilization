#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <chrono>
#include <cstddef> // Keep for size_t
#include <cstdint> // Keep for uint64_t

enum class StabilizationMode {
    FULL_LOCK,
    TRANSLATION_LOCK,
    ROTATION_LOCK,
    GLOBAL_SMOOTHING
};

struct Transformation {
    cv::Mat H;
    uint64_t from_frame_idx;
    uint64_t to_frame_idx;
    
    Transformation inverse() const {
        cv::Mat inv;
        cv::invert(H, inv);
        return Transformation{inv, to_frame_idx, from_frame_idx};
    }
};

struct Frame {
    cv::Mat image;
    uint64_t frame_idx;
};

struct StabilizationWindow {
    std::deque<Transformation> transformations;
    std::deque<Frame> frames;
    uint64_t current_frame_idx{0};
};

class Stabilizer {
public:
    Stabilizer(size_t windowSize = 15, int workingHeight = 360);
    
    // Process a frame and return the stabilized version
    cv::Mat stabilizeFrame(const cv::Mat& frame);
    
    // Reset the stabilizer state
    void reset();

private:
    size_t smoothingWindowSize_;
    int workingHeight_; // Target height for internal processing
    double scaleFactor_; // Scale factor between original and working resolution
    cv::Size originalSize_; // Original frame size
    cv::Size workingSize_; // Working frame size
    
    StabilizationWindow stabilizationWindow_; // Single instance of StabilizationWindow
    
    cv::Mat prevGray_;
    std::vector<cv::Point2f> prevPoints_;

    // Timing variables (rolling averages)
    std::chrono::duration<double, std::milli> gftt_avg_duration_ms_{0.0};
    long long gftt_call_count_{0};
    std::chrono::duration<double, std::milli> lk_avg_duration_ms_{0.0};
    long long lk_call_count_{0};
    std::chrono::duration<double, std::milli> homography_avg_duration_ms_{0.0};
    long long homography_call_count_{0};
    std::chrono::duration<double, std::milli> warp_avg_duration_ms_{0.0};
    long long warp_call_count_{0};

    // Timing print variables
    std::chrono::high_resolution_clock::time_point last_print_time_;
    const std::chrono::milliseconds print_interval_{1000}; // Print every 1000ms
};
