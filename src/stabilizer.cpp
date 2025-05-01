#include "stabilizer.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp> // For cvtColor
#include <opencv2/video/tracking.hpp> // Need this for LK
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric> // For std::accumulate
#include <deque>   // Include deque
#include <iostream> // For potential debug/error messages
#include <chrono>   // Include chrono for timing


Stabilizer::Stabilizer(size_t windowSize, int workingHeight)
    : smoothingWindowSize_(windowSize), workingHeight_(workingHeight), scaleFactor_(1.0)
{
    reset();
    last_print_time_ = std::chrono::high_resolution_clock::now();
}

void Stabilizer::reset() {
    prevGray_ = cv::Mat();
    prevPoints_.clear();
    stabilizationWindow_.transformations.clear();
    stabilizationWindow_.frames.clear();
    stabilizationWindow_.current_frame_idx = 0;
    originalSize_ = cv::Size(0, 0);
    workingSize_ = cv::Size(0, 0);
    scaleFactor_ = 1.0;

    // Reset timing variables
    gftt_avg_duration_ms_ = std::chrono::duration<double, std::milli>(0.0);
    gftt_call_count_ = 0;
    lk_avg_duration_ms_ = std::chrono::duration<double, std::milli>(0.0);
    lk_call_count_ = 0;
    homography_avg_duration_ms_ = std::chrono::duration<double, std::milli>(0.0);
    homography_call_count_ = 0;
    warp_avg_duration_ms_ = std::chrono::duration<double, std::milli>(0.0);
    warp_call_count_ = 0;
}

cv::Mat Stabilizer::stabilizeFrame(const cv::Mat& frame) {

    // Store original size if first frame or size has changed
    if (originalSize_.width != frame.cols || originalSize_.height != frame.rows) {
        originalSize_ = frame.size();
        // Calculate working size maintaining aspect ratio
        scaleFactor_ = static_cast<double>(workingHeight_) / frame.rows;
        workingSize_ = cv::Size(static_cast<int>(frame.cols * scaleFactor_), workingHeight_);
    }

    // Increment frame index
    uint64_t current_idx = 0;
    if (!stabilizationWindow_.frames.empty()) {
        current_idx = stabilizationWindow_.frames.back().frame_idx + 1;
    }
    
    // Store the current frame
    Frame currentFrame;
    currentFrame.image = frame.clone();
    currentFrame.frame_idx = current_idx;
    stabilizationWindow_.frames.push_back(currentFrame);
    
    // Keep only the necessary frames
    while (stabilizationWindow_.frames.size() > smoothingWindowSize_) {
        stabilizationWindow_.frames.pop_front();
    }

    // Resize the input frame to working resolution
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, workingSize_, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(resizedFrame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat T_stabilize = cv::Mat::eye(3, 3, CV_64F);
    const size_t MIN_RELIABLE_POINTS = 10;
    const int MAX_FEATURES_TO_DETECT = 200;
    const double QUALITY_LEVEL = 0.01;
    const int MIN_DISTANCE = static_cast<int>(30 * scaleFactor_); // Scale min distance proportionally

    if (prevGray_.empty()) {
        auto start_gftt = std::chrono::high_resolution_clock::now();
        cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
        auto end_gftt = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_gftt - start_gftt);
        gftt_call_count_++;
        gftt_avg_duration_ms_ += (duration_ms - gftt_avg_duration_ms_) / gftt_call_count_;

        gray.copyTo(prevGray_);
        return frame;
    }

    // --- Track Features
    std::vector<cv::Point2f> currPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    size_t tracked_count = 0;

    if (!prevPoints_.empty()) {
        auto start_lk = std::chrono::high_resolution_clock::now();
        cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPoints_, currPoints, status, err);
        auto end_lk = std::chrono::high_resolution_clock::now();
        auto duration_ms_lk = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_lk - start_lk);
        lk_call_count_++;
        lk_avg_duration_ms_ += (duration_ms_lk - lk_avg_duration_ms_) / lk_call_count_;

        // --- Filter Points
        size_t i, k;
        for (i = k = 0; i < status.size(); ++i) {
            if (status[i]) {
                // In-place filtering
                if (k < i) { // Avoid self-assignment if k==i
                     // Check bounds before assignment (important for safety)
                     if (k < prevPoints_.size() && k < currPoints.size() && i < prevPoints_.size() && i < currPoints.size()) {
                         prevPoints_[k] = prevPoints_[i];
                         currPoints[k] = currPoints[i];
                     } else {
                         // Should not happen if LK returns consistent sizes, but handle defensively
                         std::cerr << "Warning: Index out of bounds during point filtering." << std::endl;
                         continue;
                     }
                } else if (k >= prevPoints_.size() || k >= currPoints.size()) {
                     // Defensive check
                     std::cerr << "Warning: Index 'k' out of bounds during point filtering." << std::endl;
                     continue;
                }
                 k++;
            }
        }
        // Resize vectors to contain only the tracked points
        if (k < prevPoints_.size()) prevPoints_.resize(k);
        if (k < currPoints.size()) currPoints.resize(k);
        tracked_count = k;

    } // else: tracked_count remains 0 if prevPoints_ was empty

    // --- Compute H_k,k-1 (H_curr_prev) using points ---
    cv::Mat H_curr_prev = cv::Mat::eye(3, 3, CV_64F);
    bool H_computed_reliably = false;

    if (tracked_count >= MIN_RELIABLE_POINTS) {
        auto start_motion_estimation = std::chrono::high_resolution_clock::now();
        // Estimate Euclidean transform (rigid + isotropic scale) instead of full homography.
        // This is often more robust for shaky video or when perspective distortion is minimal.
        cv::Mat M = cv::estimateAffinePartial2D(prevPoints_, currPoints, cv::noArray(), cv::RANSAC); // function with unfortunate name. This is actually Euclidean transform estimation.
        auto end_motion_estimation = std::chrono::high_resolution_clock::now();
        auto duration_ms_motion = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_motion_estimation - start_motion_estimation);
        homography_call_count_++;
        homography_avg_duration_ms_ += (duration_ms_motion - homography_avg_duration_ms_) / homography_call_count_;

        if (!M.empty() && M.rows == 2 && M.cols == 3 && cv::checkRange(M)) {
             // Convert 2x3 matrix M = [sR | t] to 3x3 homography H
             // H = [ M_row1 ]
             //     [ M_row2 ]
             //     [ 0  0  1 ]
             H_curr_prev = cv::Mat::eye(3, 3, CV_64F);
             M.copyTo(H_curr_prev(cv::Rect(0, 0, 3, 2)));
             H_computed_reliably = true;
        }
    }
    // If tracking failed or transform invalid, H_curr_prev remains Identity

    // --- Update transformations in the StabilizationWindow ---
    Transformation current_transform;
    current_transform.H = H_curr_prev.clone();

    assert(current_idx >= 1);
    current_transform.from_frame_idx = current_idx - 1;
    current_transform.to_frame_idx = current_idx;
    stabilizationWindow_.transformations.push_back(current_transform);

    // Keep only necessary transformations
    while (stabilizationWindow_.transformations.size() > smoothingWindowSize_ - 1) {
        stabilizationWindow_.transformations.pop_front();
    }
    
    assert(stabilizationWindow_.transformations.size() == stabilizationWindow_.frames.size() - 1);

    // --- Calculate Stabilization Transform T_stabilize ---
    if (stabilizationWindow_.transformations.size() == smoothingWindowSize_ - 1) {
        cv::Mat H_curr_oldest = cv::Mat::eye(3, 3, CV_64F);
        for (int i = stabilizationWindow_.transformations.size() - 1; i >= 0; --i) {
            H_curr_oldest = H_curr_oldest * stabilizationWindow_.transformations[i].H;
        }

        cv::Mat T_inv = H_curr_oldest.inv();
        if (!T_inv.empty() && cv::checkRange(T_inv)) {
             T_stabilize = T_inv;
        }
    }

    // --- Scale the transform matrix to original resolution ---
    cv::Mat T_stabilize_scaled = T_stabilize.clone();
    if (scaleFactor_ != 1.0) {
        // Adjust translation components
        T_stabilize_scaled.at<double>(0, 2) /= scaleFactor_;
        T_stabilize_scaled.at<double>(1, 2) /= scaleFactor_;
    }

    // --- Warp the original frame ---
    cv::Mat stabilized;
    if (!T_stabilize_scaled.empty() && cv::checkRange(T_stabilize_scaled)) {
        auto start_warp = std::chrono::high_resolution_clock::now();
        cv::warpPerspective(frame, stabilized, T_stabilize_scaled, frame.size());
        auto end_warp = std::chrono::high_resolution_clock::now();
        auto duration_ms_warp = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_warp - start_warp);
        warp_call_count_++;
        warp_avg_duration_ms_ += (duration_ms_warp - warp_avg_duration_ms_) / warp_call_count_;
    } else {
         frame.copyTo(stabilized);
    }

    // --- Prepare for Next Frame
    // Always detect features on the current frame using GFTT
    auto start_gftt = std::chrono::high_resolution_clock::now();
    cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
    auto end_gftt = std::chrono::high_resolution_clock::now();
    auto duration_ms_gftt = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_gftt - start_gftt);
    gftt_call_count_++;
    gftt_avg_duration_ms_ += (duration_ms_gftt - gftt_avg_duration_ms_) / gftt_call_count_;
    
    gray.copyTo(prevGray_); // Update previous gray image

    // --- Print Timings Periodically ---
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_since_print = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time_);
    if (elapsed_since_print >= print_interval_) {
        std::cout << "--- Timing Averages (ms) ---" << std::endl;
        std::cout << "  Working resolution: " << workingSize_.width << "x" << workingSize_.height 
                  << " (scale: " << scaleFactor_ << ")" << std::endl;
        if (gftt_call_count_ > 0) {
            std::cout << "  goodFeaturesToTrack: " << gftt_avg_duration_ms_.count()
                      << " ms (calls: " << gftt_call_count_ << ")" << std::endl;
        }
        if (lk_call_count_ > 0) {
            std::cout << "  calcOpticalFlowPyrLK: " << lk_avg_duration_ms_.count()
                      << " ms (calls: " << lk_call_count_ << ")" << std::endl;
        }
        if (homography_call_count_ > 0) {
            std::cout << "  estimateAffinePartial2D: " << homography_avg_duration_ms_.count()
                      << " ms (calls: " << homography_call_count_ << ")" << std::endl;
        }
        if (warp_call_count_ > 0) {
             std::cout << "  warpPerspective: " << warp_avg_duration_ms_.count()
                       << " ms (calls: " << warp_call_count_ << ")" << std::endl;
        }
        std::cout << "----------------------------" << std::endl;
        last_print_time_ = now;
    }

    return stabilized;
}
