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

struct HomographyParameters {
    double s;       // scaling factor
    double theta;   // in-image-plane rotation angle
    double k;       // anisotropy scaling factor
    double delta;   // shear factor
    cv::Vec2d t;  // translation vector
    cv::Vec2d v;  // horizon line shift vector
};

struct Transformation {
    cv::Mat H;
    size_t from_frame_idx;
    size_t to_frame_idx;
    
    Transformation inverse() const {
        cv::Mat inv;
        cv::invert(H, inv);
        return Transformation{inv, to_frame_idx, from_frame_idx};
    }
};

struct Frame {
    cv::Mat image;
    size_t frame_idx;
};

struct StabilizationWindow {
    std::deque<Transformation> transformations;
    std::deque<Frame> frames;
};

class Stabilizer {
public:
    // Type alias for millisecond duration to simplify code
    using milli_duration = std::chrono::duration<double, std::milli>;
    
    // Helper to get current time more concisely
    inline auto now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    Stabilizer(size_t pastFrames = 15, size_t futureFrames = 15, int workingHeight = 360);
    
    // Process a frame and return the stabilized version
    cv::Mat stabilizeFrame(const cv::Mat& frame);
    
    // Reset the stabilizer state
    void reset();

    // Set stabilization mode
    void setStabilizationMode(StabilizationMode mode);

    inline size_t totalFrameWindowSize() const {
        return totalPastFrames_ + 1 + totalFutureFrames_;
    }

    // Helper method to convert duration to milliseconds
    inline milli_duration toMilliseconds(const std::chrono::high_resolution_clock::time_point& start,
                                      const std::chrono::high_resolution_clock::time_point& end) {
        return std::chrono::duration_cast<milli_duration>(end - start);
    }
    
    // Decomposes a homography H into H = H_Similarity * H_Shear * H_Projective
    static HomographyParameters decomposeHomography(const cv::Mat& H);

private:
    // Helper methods for stabilizeFrame
    void initializeFrame(const cv::Mat& frame);
    cv::Mat prepareTrailBackground();
    void addFrameToWindow(const cv::Mat& frame);
    std::pair<std::vector<cv::Point2f>, bool> trackFeatures(const cv::Mat& gray);
    cv::Mat estimateMotion(const std::vector<cv::Point2f>& currPoints, bool reliable);
    void updateTransformations(const cv::Mat& H_prev2curr, uint64_t current_idx);
    cv::Mat calculateFullLockStabilization(const Transformation& current_transform);
    cv::Mat calculateGlobalSmoothingStabilization(size_t presentation_frame_idx);
    cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& H_stabilize_scaled, 
                      const cv::Mat& next_trail_background, size_t presentation_frame_idx);
    void detectNewFeatures(const cv::Mat& gray);
    void printTimings();
    
    size_t totalPastFrames_;
    size_t totalFutureFrames_;
    int workingHeight_; // Target height for internal processing
    double scaleFactor_; // Scale factor between original and working resolution
    cv::Size originalSize_; // Original frame size
    cv::Size workingSize_; // Working frame size
    
    StabilizationWindow stabilizationWindow_; // Single instance of StabilizationWindow
    
    cv::Mat prevGray_;
    std::vector<cv::Point2f> prevPoints_;

    cv::Mat trail_background_; // Background for the trail effect

    Transformation accumulatedTransform_; // Store the accumulated transformation for full lock mode

    // Timing variables (rolling averages)
    milli_duration gftt_avg_duration_ms_{0.0};
    long long gftt_call_count_{0};
    milli_duration lk_avg_duration_ms_{0.0};
    long long lk_call_count_{0};
    milli_duration homography_avg_duration_ms_{0.0};
    long long homography_call_count_{0};
    milli_duration warp_avg_duration_ms_{0.0};
    long long warp_call_count_{0};
    StabilizationMode stabilizationMode_{StabilizationMode::GLOBAL_SMOOTHING};

    // Timing print variables
    std::chrono::high_resolution_clock::time_point last_print_time_;
    const std::chrono::milliseconds print_interval_{1000}; // Print every 1000ms
};
