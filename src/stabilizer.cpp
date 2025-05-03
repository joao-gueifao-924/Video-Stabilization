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


Stabilizer::Stabilizer(size_t pastFrames, size_t futureFrames, int workingHeight)
    : totalPastFrames_(pastFrames), totalFutureFrames_(futureFrames), 
      workingHeight_(workingHeight), scaleFactor_(1.0)
{
    reset();
    last_print_time_ = now();
}

void Stabilizer::reset() {
    prevGray_ = cv::Mat();
    prevPoints_.clear();
    stabilizationWindow_.transformations.clear();
    stabilizationWindow_.frames.clear();
    originalSize_ = cv::Size(0, 0);
    workingSize_ = cv::Size(0, 0);
    scaleFactor_ = 1.0;

    // Reset timing variables
    gftt_avg_duration_ms_ = milli_duration(0.0);
    gftt_call_count_ = 0;
    lk_avg_duration_ms_ = milli_duration(0.0);
    lk_call_count_ = 0;
    homography_avg_duration_ms_ = milli_duration(0.0);
    homography_call_count_ = 0;
    warp_avg_duration_ms_ = milli_duration(0.0);
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
    
    // Keep only the necessary frames - discard the oldest frames
    while (stabilizationWindow_.frames.size() > totalFrameWindowSize()) {
        stabilizationWindow_.frames.pop_front();
    }

    // Resize the input frame to working resolution
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, workingSize_, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(resizedFrame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat H_stabilize = cv::Mat::eye(3, 3, CV_64F);
    const size_t MIN_RELIABLE_POINTS = 10;
    const int MAX_FEATURES_TO_DETECT = 200;
    const double QUALITY_LEVEL = 0.01;
    const int MIN_DISTANCE = static_cast<int>(30 * scaleFactor_);

    // --- First ever Frame in the video ---
    if (prevGray_.empty()) { // run only once
        auto start_gftt = now();
        cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
        auto end_gftt = now();
        auto duration_ms = toMilliseconds(start_gftt, end_gftt);
        gftt_call_count_++;
        gftt_avg_duration_ms_ += (duration_ms - gftt_avg_duration_ms_) / gftt_call_count_;

        gray.copyTo(prevGray_);
        return frame;
    }

    // --- Track Features in images
    std::vector<cv::Point2f> currPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    size_t tracked_count = 0;

    if (!prevPoints_.empty()) {
        auto start_lk = now();
        cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPoints_, currPoints, status, err);
        auto end_lk = now();
        auto duration_ms_lk = toMilliseconds(start_lk, end_lk);
        lk_call_count_++;
        lk_avg_duration_ms_ += (duration_ms_lk - lk_avg_duration_ms_) / lk_call_count_;

        // --- Filter Points
        size_t i, k;
        for (i = k = 0; i < status.size(); ++i) {
            if (status[i]) {
                // In-place filtering
                if (k < i) { // Avoid self-assignment if k==i
                     // Check bounds before assignment (important for safety)
                     if (k < prevPoints_.size() && k < currPoints.size() && i < prevPoints_.size() 
                                                                        && i < currPoints.size()) {
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

    // --- Compute transformation from previous frame to current frame using tracked points ---
    cv::Mat H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
    bool H_computed_reliably = false;

    if (tracked_count >= MIN_RELIABLE_POINTS) {
        auto start_motion_estimation = now();
        // Estimate Euclidean transform (rigid + isotropic scale) instead of full homography.
        // This is often more robust for shaky video or when perspective distortion is minimal.
        cv::Mat M = cv::estimateAffinePartial2D(prevPoints_, currPoints, cv::noArray(), cv::RANSAC);
        auto end_motion_estimation = now();
        
        auto duration_ms_motion = toMilliseconds(start_motion_estimation, end_motion_estimation);
                
        homography_call_count_++;
        homography_avg_duration_ms_ += 
            (duration_ms_motion - homography_avg_duration_ms_) / homography_call_count_;

        if (!M.empty() && M.rows == 2 && M.cols == 3 && cv::checkRange(M)) {
             // Convert 2x3 matrix M = [sR | t] to 3x3 homography H
             // H = [ M_row1 ]
             //     [ M_row2 ]
             //     [ 0  0  1 ]
             H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
             M.copyTo(H_prev2curr(cv::Rect(0, 0, 3, 2)));
             H_computed_reliably = true;
        }
    }
    // If tracking failed or transform invalid, H_prev2curr remains Identity

    // --- Update transformations in the StabilizationWindow ---
    Transformation current_transform;
    current_transform.H = H_prev2curr.clone();

    assert(current_idx >= 1);
    current_transform.from_frame_idx = current_idx - 1;
    current_transform.to_frame_idx = current_idx;
    stabilizationWindow_.transformations.push_back(current_transform);

    // Keep only necessary transformations
    while (stabilizationWindow_.transformations.size() > totalFrameWindowSize() -1) {
        stabilizationWindow_.transformations.pop_front();
    }
    
    assert(stabilizationWindow_.transformations.size() == stabilizationWindow_.frames.size() - 1);

    // --- Calculate Stabilization Transform H_stabilize ---

    // How it works:
    // From frame to be presented next to the user (presentation_frame), compute a corrective transformation that is
    // the average of all tranformations in the stabilization window between presentation_frame and 
    // each of the previous frames and between presentation_frame and each of the future frames.
    // Remember that presentation_frame is somewhere in the middle of the stabilization window.
    // This is a simple and effective way to stabilize the video by correcting for camera movement.
    // Especially during premature execution of the program or after a stabilization reset() operation, 
    // the total frames inserted in the stabilization window might be less than the totalWindowSize().
    // We accomodate for this dynamic total of available frames for stabilization.

    // Initialize transformation average and accumulator
    cv::Mat H_avg = cv::Mat::zeros(3, 3, CV_64F);
    int count = 0;
    
    // Initialize cumulative transformation matrix (identity)
    cv::Mat H_accum = cv::Mat::eye(3, 3, CV_64F);

    // Find presentation_frame, given totalPastFrames_, totalFutureFrames_ and the current number
    // of frames and transformations available in the stabilizationWindow_.
    // If the number of available frames in stabilizationWindow_ is totalFutureFrames_ or less, presentation_frame will be
    // the frame most to the left (oldest) and use all frames to its right to compute H_avg.
    // When the number of available frames in stabilizationWindow_ becomes bigger than totalFutureFrames_, presentation_frame will be
    // somewhere in the middle of stabilizationWindow_, depending on how many past frames are available to the left of presentation_frame.
    size_t presentation_frame_idx = 0;
    if (stabilizationWindow_.frames.size() > totalFutureFrames_) {
        presentation_frame_idx = stabilizationWindow_.frames.size() - totalFutureFrames_ - 1;
    }
    
    // Calculate all transformations from presentation_frame to each of the older frames, if any
    for (int i = presentation_frame_idx; i > 0; --i) {
        // Update cumulative transformation
        size_t transformation_idx = i - 1;
        // the transformation is defined from previous frame to next frame, hence the inverse
        Transformation T_inv = stabilizationWindow_.transformations[transformation_idx].inverse();
        H_accum = T_inv.H * H_accum; // left matrix multiplication

        // Add to average
        H_avg += H_accum;
        count++;
    }

    // Reinitialize cumulative transformation matrix (identity)
    H_accum = cv::Mat::eye(3, 3, CV_64F);

    // Calculate all transformations from presentation_frame to each of the newer frames, if any.
    for (int i = presentation_frame_idx; i < stabilizationWindow_.transformations.size() - 1; ++i) {
        // Update cumulative transformation
        size_t transformation_idx = i;
        // the transformation is defined from previous frame to next frame, hence no need to invert here
        Transformation T_inv = stabilizationWindow_.transformations[transformation_idx];
        H_accum = H_accum * T_inv.H; // right matrix multiplication

        // Add to average
        H_avg += H_accum;
        count++;
    }
        
    // Calculate average transformation
    if (count > 0) {
        H_avg = H_avg / count;
        
        // Check if average transformation is valid
        if (!H_avg.empty() && cv::checkRange(H_avg)) {
            H_stabilize = H_avg;
        }
    }

    // --- Scale the transform matrix to original resolution ---
    cv::Mat H_stabilize_scaled = H_stabilize.clone();
    if (scaleFactor_ != 1.0) {
        // Adjust translation components
        H_stabilize_scaled.at<double>(0, 2) /= scaleFactor_;
        H_stabilize_scaled.at<double>(1, 2) /= scaleFactor_;
    }

    // --- Warp the original frame ---
    cv::Mat stabilized;
    if (!H_stabilize_scaled.empty() && cv::checkRange(H_stabilize_scaled)) {
        auto start_warp = now();
        
        cv::Mat presentation_image = stabilizationWindow_.frames[presentation_frame_idx].image;
        cv::warpPerspective(presentation_image, stabilized, H_stabilize_scaled, frame.size());
        auto end_warp = now();
        
        auto duration_ms_warp = toMilliseconds(start_warp, end_warp);
        
        warp_call_count_++;
        warp_avg_duration_ms_ += (duration_ms_warp - warp_avg_duration_ms_) / warp_call_count_;
    } else {
         frame.copyTo(stabilized);
    }

    // --- Prepare for Next Frame
    // Always detect features on the current frame using GFTT
    auto start_gftt = now();
    cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
    auto end_gftt = now();
    auto duration_ms_gftt = toMilliseconds(start_gftt, end_gftt);
    gftt_call_count_++;
    gftt_avg_duration_ms_ += (duration_ms_gftt - gftt_avg_duration_ms_) / gftt_call_count_;
    
    gray.copyTo(prevGray_); // Update previous gray image

    // --- Print Timings Periodically ---
    auto elapsed_since_print = milli_duration(now() - last_print_time_);
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
        last_print_time_ = now();
    }

    return stabilized;
}
