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


Stabilizer::Stabilizer(size_t windowSize)
    : smoothingWindowSize_(windowSize)
{
    if (smoothingWindowSize_ < 2) smoothingWindowSize_ = 2;
    reset();
    last_print_time_ = std::chrono::high_resolution_clock::now();
}

void Stabilizer::reset() {
    prevGray_ = cv::Mat();
    prevPoints_.clear();
    H_window_.clear();

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
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat T_stabilize = cv::Mat::eye(3, 3, CV_64F);
    const size_t MIN_RELIABLE_POINTS = 10;
    const int MAX_FEATURES_TO_DETECT = 200;

    if (prevGray_.empty()) {
        auto start_gftt = std::chrono::high_resolution_clock::now();
        cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, 0.01, 30);
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
        // Note: Reusing homography timing variables for simplicity, rename if needed.
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

    // --- Update Rolling Window ---
    H_window_.push_back(H_curr_prev);
    while (H_window_.size() > smoothingWindowSize_ - 1) {
        H_window_.pop_front();
    }

    // --- Calculate Stabilization Transform T_stabilize ---
    if (H_window_.size() == smoothingWindowSize_ - 1) {
        cv::Mat H_curr_oldest = cv::Mat::eye(3, 3, CV_64F);
        for (int i = H_window_.size() - 1; i >= 0; --i) {
            H_curr_oldest = H_curr_oldest * H_window_[i];
        }

        cv::Mat T_inv = H_curr_oldest.inv();
        if (!T_inv.empty() && cv::checkRange(T_inv)) {
             T_stabilize = T_inv;
        }
    }

    // --- Warp ---
    cv::Mat stabilized;
    if (!T_stabilize.empty() && cv::checkRange(T_stabilize)) {
        auto start_warp = std::chrono::high_resolution_clock::now();
        cv::warpPerspective(frame, stabilized, T_stabilize, frame.size());
        auto end_warp = std::chrono::high_resolution_clock::now();
        auto duration_ms_warp = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_warp - start_warp);
        warp_call_count_++;
        warp_avg_duration_ms_ += (duration_ms_warp - warp_avg_duration_ms_) / warp_call_count_;
    } else {
         frame.copyTo(stabilized);
    }

    // --- Prepare for Next Frame
    if (H_computed_reliably) {
        // Use the successfully tracked points from LK
        prevPoints_ = currPoints;
    } else {
        // Re-detect features on the current frame using GFTT
        auto start_gftt = std::chrono::high_resolution_clock::now();
        cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, 0.01, 30);
        auto end_gftt = std::chrono::high_resolution_clock::now();
        auto duration_ms_gftt = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_gftt - start_gftt);
        gftt_call_count_++;
        gftt_avg_duration_ms_ += (duration_ms_gftt - gftt_avg_duration_ms_) / gftt_call_count_;
    }
    gray.copyTo(prevGray_); // Update previous gray image

    // --- Print Timings Periodically ---
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed_since_print = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time_);
    if (elapsed_since_print >= print_interval_) {
        std::cout << "--- Timing Averages (ms) ---" << std::endl;
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
