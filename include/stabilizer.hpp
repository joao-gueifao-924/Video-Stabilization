#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <chrono>
#include <cstddef> // Keep for size_t
#include <cstdint> // Keep for uint64_t

enum class StabilizationMode {
    ACCUMULATED_FULL_LOCK,
    ORB_FULL_LOCK,
    SIFT_FULL_LOCK,
    TRANSLATION_LOCK,
    ROTATION_LOCK,
    GLOBAL_SMOOTHING
};

struct HomographyParameters {
    double s{1};       // scaling factor
    double theta{0};   // in-image-plane rotation angle
    //cv::Point2d c;  // center of scaling and rotation
    double k{0};       // anisotropy scaling factor
    double delta{0};   // shear factor
    cv::Vec2d t{0,0};    // translation vector
    cv::Vec2d v{0,0};    // horizon line shift vector

    // Print to terminal for debugging:
    std::string print() const {
        return "s: " + std::to_string(s) + ", theta: " + std::to_string(theta) + ", k: " + std::to_string(k) + ", delta: " + std::to_string(delta) + ", t: " + std::to_string(t[0]) + ", " + std::to_string(t[1]) + ", v: " + std::to_string(v[0]) + ", " + std::to_string(v[1]);
    }
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
    
    static bool decomposeHomography(const cv::Mat& H, HomographyParameters &params_out, cv::Point2d rot_center = cv::Point2d(0,0));
    static cv::Mat composeHomography(const HomographyParameters& params, cv::Point2d rot_center = cv::Point2d(0,0));

    static cv::Mat copyFeathered(const cv::Mat foreground, const cv::Mat background_image, const cv::Mat H);

private:
    // Helper methods for stabilizeFrame
    void initializeFrame(const cv::Mat& frame);
    cv::Mat prepareTrailBackground();
    void addFrameToWindow(const cv::Mat& frame);
    
    void trackFeatures(const cv::Mat& previous_gray, const cv::Mat& current_gray, 
                        const std::vector<cv::Point2f> & previousPoints,
                        std::vector<cv::Point2f> & filtered_previousPoints,
                        std::vector<cv::Point2f> & filtered_currentPoints);

    cv::Mat estimateMotion(const std::vector<cv::Point2f>& previousPoints, const std::vector<cv::Point2f>& currentPoints);
    
    void updateTransformations(const cv::Mat& H_prev2curr, uint64_t current_idx);
    
    cv::Mat calculateFullLockStabilization(size_t presentation_frame_idx);
    
    cv::Mat calculateGlobalSmoothingStabilization(size_t presentation_frame_idx);
    
    cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& H_stabilize_scaled, 
                      const cv::Mat& next_trail_background, size_t presentation_frame_idx);
    
    std::vector<cv::Point2f> detectNewFeatures(const cv::Mat& gray, cv::Mat mask = cv::Mat());
    
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

    // ORB-based motion estimation variables
    int64_t referenceFrameIdx_; // Reference frame index for full lock mode
    cv::Mat referenceGray_; // Reference frame grayscale image
    std::vector<cv::KeyPoint> orb_referenceKeypoints_; // Reference frame keypoints
    cv::Mat orb_referenceDescriptors_; // Reference frame descriptors


    std::vector<cv::KeyPoint> sift_referenceKeypoints_; // Reference frame keypoints
    cv::Mat sift_referenceDescriptors_; // Reference frame descriptors

    cv::Ptr<cv::ORB> orb_; // ORB feature detector
    cv::Ptr<cv::SIFT> sift_; // SIFT feature detector

    std::vector<cv::Point2f> gftt_reference_corners_;

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
