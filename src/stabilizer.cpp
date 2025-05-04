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
      workingHeight_(workingHeight)
{
    reset();
    last_print_time_ = now();
}

void Stabilizer::reset() {
    prevGray_ = cv::Mat();
    prevPoints_.clear();
    trail_background_ = cv::Mat();
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
    accumulatedTransform_ = Transformation();
    stabilizationMode_ = StabilizationMode::GLOBAL_SMOOTHING;
}

void Stabilizer::setStabilizationMode(StabilizationMode mode) {
    if (stabilizationMode_ != mode) {
        // Reset the accumulated transform when switching to any mode
        accumulatedTransform_ = Transformation();

        stabilizationMode_ = mode;
        std::cout << "Stabilization mode changed to: ";
        switch (mode) {
            case StabilizationMode::FULL_LOCK:
                std::cout << "FULL_LOCK";
                break;
            case StabilizationMode::TRANSLATION_LOCK:
                std::cout << "TRANSLATION_LOCK";
                break;
            case StabilizationMode::ROTATION_LOCK:
                std::cout << "ROTATION_LOCK";
                break;
            case StabilizationMode::GLOBAL_SMOOTHING:
                std::cout << "GLOBAL_SMOOTHING";
                break;
        }
        std::cout << std::endl;
    }
}

void Stabilizer::initializeFrame(const cv::Mat& frame) {
    // Store original size if first frame or size has changed
    if (originalSize_.width != frame.cols || originalSize_.height != frame.rows) {
        originalSize_ = frame.size();
        // Calculate working size maintaining aspect ratio
        scaleFactor_ = static_cast<double>(workingHeight_) / frame.rows;
        workingSize_ = cv::Size(static_cast<int>(frame.cols * scaleFactor_), workingHeight_);
    }
    
    // Initialize or resize trail background
    if (trail_background_.empty() || trail_background_.size() != originalSize_) {
        trail_background_ = cv::Mat::zeros(originalSize_, frame.type());
    }
}

cv::Mat Stabilizer::prepareTrailBackground() {
    cv::Mat next_trail_background;
    // Grayscale
    cv::cvtColor(trail_background_, next_trail_background, cv::COLOR_BGR2GRAY);
    // Convert back to BGR for compatibility
    cv::cvtColor(next_trail_background, next_trail_background, cv::COLOR_GRAY2BGR);
    cv::GaussianBlur(next_trail_background, next_trail_background, cv::Size(3, 3), 0);
    // Darken
    next_trail_background *= 0.99; 
    return next_trail_background;
}

void Stabilizer::addFrameToWindow(const cv::Mat& frame) {
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
}

std::pair<std::vector<cv::Point2f>, bool> Stabilizer::trackFeatures(const cv::Mat& gray) {
    const size_t MIN_RELIABLE_POINTS = 10;
    std::vector<cv::Point2f> currPoints;
    bool featuresTracked = false;
    size_t tracked_count = 0;

    if (!prevPoints_.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        
        auto start_lk = now();
        cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPoints_, currPoints, status, err);
        auto end_lk = now();
        auto duration_ms_lk = toMilliseconds(start_lk, end_lk);
        lk_call_count_++;
        lk_avg_duration_ms_ += (duration_ms_lk - lk_avg_duration_ms_) / lk_call_count_;

        // Filter Points
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
        
        featuresTracked = (tracked_count >= MIN_RELIABLE_POINTS);
    }
    
    return {currPoints, featuresTracked};
}

cv::Mat Stabilizer::estimateMotion(const std::vector<cv::Point2f>& currPoints, bool reliable) {
    cv::Mat H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
    
    if (reliable) {
        auto start_motion_estimation = now();
        // Estimate Euclidean transform (rigid + isotropic scale) instead of full homography.
        //cv::Mat M = cv::findHomography(prevPoints_, currPoints, cv::RANSAC);
        cv::Mat M = cv::estimateAffinePartial2D(prevPoints_, currPoints, cv::noArray(), cv::RANSAC);
        auto end_motion_estimation = now();
        
        auto duration_ms_motion = toMilliseconds(start_motion_estimation, end_motion_estimation);
                
        homography_call_count_++;
        homography_avg_duration_ms_ += 
            (duration_ms_motion - homography_avg_duration_ms_) / homography_call_count_;

        if (!M.empty() && cv::checkRange(M)) {
            if (M.rows == 2 && M.cols == 3) {
                // Convert 2x3 matrix M = [sR | t] to 3x3 homography H
                H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
                M.copyTo(H_prev2curr(cv::Rect(0, 0, 3, 2)));
            } else { // M is already a 3x3 homography
                M.copyTo(H_prev2curr);
            }
        }
    }
    
    return H_prev2curr;
}

void Stabilizer::updateTransformations(const cv::Mat& H_prev2curr, uint64_t current_idx) {
    Transformation current_transform;
    current_transform.H = H_prev2curr.clone();
    current_transform.from_frame_idx = current_idx - 1;
    current_transform.to_frame_idx = current_idx;
    stabilizationWindow_.transformations.push_back(current_transform);

    // Keep only necessary transformations
    while (stabilizationWindow_.transformations.size() > totalFrameWindowSize() - 1) {
        stabilizationWindow_.transformations.pop_front();
    }
}

cv::Mat Stabilizer::calculateFullLockStabilization(size_t presentation_frame_idx) {
    // If we are not using any lock, we can just return the identity matrix
    if (stabilizationMode_ == StabilizationMode::GLOBAL_SMOOTHING) {
        return cv::Mat::eye(3, 3, CV_64F);
    }

    size_t frame_idx = stabilizationWindow_.frames[presentation_frame_idx].frame_idx;

    // If this iteration is the first time we are computing the stabilization transform, use the identity matrix
    if (accumulatedTransform_.H.empty()) {
        accumulatedTransform_.H = cv::Mat::eye(3, 3, CV_64F);
        accumulatedTransform_.from_frame_idx = frame_idx;
        accumulatedTransform_.to_frame_idx = frame_idx;
    }
    else {
        Transformation next_transform = stabilizationWindow_.transformations[presentation_frame_idx -1];
        assert(next_transform.from_frame_idx == accumulatedTransform_.to_frame_idx);

        // update the accumulated transform by multiplying it with the transformation
        accumulatedTransform_.H = accumulatedTransform_.H * next_transform.H;
        accumulatedTransform_.to_frame_idx = next_transform.to_frame_idx;
    }

    // Use the accumulated transform to stabilize the frame
    return accumulatedTransform_.H.inv();
}

cv::Mat Stabilizer::calculateGlobalSmoothingStabilization(size_t presentation_frame_idx) {
    cv::Mat H_stabilize = cv::Mat::eye(3, 3, CV_64F);
    
    // Initialize transformation average and accumulator
    cv::Mat H_avg = cv::Mat::zeros(3, 3, CV_64F);
    int count = 0;
    
    // Initialize cumulative transformation matrix (identity)
    cv::Mat H_accum = cv::Mat::eye(3, 3, CV_64F);

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

    return H_stabilize;
}

cv::Mat Stabilizer::warpFrame(const cv::Mat& frame, const cv::Mat& H_stabilize_scaled, const cv::Mat& next_trail_background, size_t presentation_frame_idx) {
    cv::Mat stabilized = next_trail_background.clone();
    
    if (!H_stabilize_scaled.empty() && cv::checkRange(H_stabilize_scaled)) {
        auto start_warp = now();
        
        cv::Mat presentation_image = stabilizationWindow_.frames[presentation_frame_idx].image;
        cv::Mat warped_image;
        // Warp into a separate Mat
        cv::warpPerspective(presentation_image, warped_image, H_stabilize_scaled, frame.size());
        
        // Create a mask by transforming the frame corners
        const int BORDER_SIZE = 10;
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        std::vector<cv::Point2f> original_corners(4);
        original_corners[0] = cv::Point2f(BORDER_SIZE, BORDER_SIZE);
        original_corners[1] = cv::Point2f(presentation_image.cols - BORDER_SIZE, BORDER_SIZE);
        original_corners[2] = cv::Point2f(presentation_image.cols - BORDER_SIZE, presentation_image.rows - BORDER_SIZE);
        original_corners[3] = cv::Point2f(BORDER_SIZE, presentation_image.rows - BORDER_SIZE);
        
        std::vector<cv::Point2f> transformed_corners(4);
        cv::perspectiveTransform(original_corners, transformed_corners, H_stabilize_scaled);
        
        // Convert Point2f to Point for fillConvexPoly
        std::vector<cv::Point> poly_corners(4);
        for(int i = 0; i < 4; ++i) {
            poly_corners[i] = transformed_corners[i];
        }
        
        // Draw the filled polygon on the mask
        cv::fillConvexPoly(mask, poly_corners, cv::Scalar(255));

        // Copy warped image onto the trail background using the mask
        warped_image.copyTo(stabilized, mask);

        auto end_warp = now();
        
        auto duration_ms_warp = toMilliseconds(start_warp, end_warp);
        
        warp_call_count_++;
        warp_avg_duration_ms_ += (duration_ms_warp - warp_avg_duration_ms_) / warp_call_count_;
    } else {
        frame.copyTo(stabilized);
    }
    
    return stabilized;
}

void Stabilizer::detectNewFeatures(const cv::Mat& gray) {
    const int MAX_FEATURES_TO_DETECT = 200;
    const double QUALITY_LEVEL = 0.01;
    const int MIN_DISTANCE = static_cast<int>(30 * scaleFactor_);
    
    auto start_gftt = now();
    cv::goodFeaturesToTrack(gray, prevPoints_, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
    auto end_gftt = now();
    auto duration_ms_gftt = toMilliseconds(start_gftt, end_gftt);
    gftt_call_count_++;
    gftt_avg_duration_ms_ += (duration_ms_gftt - gftt_avg_duration_ms_) / gftt_call_count_;
    
    gray.copyTo(prevGray_); // Update previous gray image
}

void Stabilizer::printTimings() {
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
}

cv::Vec2d removeRotationCenterFromTranslation(const cv::Vec2d& t, const cv::Vec2d& c, double s, double theta) {
    
    theta = -theta;
    cv::Mat I = cv::Mat::eye(2, 2, CV_64F);
    cv::Mat R = (cv::Mat_<double>(2, 2) << 
                 std::cos(theta), -std::sin(theta),
                 std::sin(theta), std::cos(theta));
    cv::Vec2d t_shift = s * cv::Vec2d(cv::Mat((I - R) * cv::Vec2d(c)));
    cv::Vec2d t_shifted = cv::Vec2d(t) - 1.0 * t_shift;
    return t_shifted;
}

cv::Mat Stabilizer::stabilizeFrame(const cv::Mat& frame) {
    // Initialize frame processing
    initializeFrame(frame);
    
    // Prepare trail background for this frame
    cv::Mat next_trail_background = prepareTrailBackground();
    
    // Add frame to window and manage window size
    addFrameToWindow(frame);
    
    // Resize the input frame to working resolution
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, workingSize_, 0, 0, cv::INTER_LINEAR);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(resizedFrame, gray, cv::COLOR_BGR2GRAY);

    // If first frame, initialize and return
    if (prevGray_.empty()) {
        detectNewFeatures(gray);
        return frame;
    }
    
    // Track features between frames
    auto [currPoints, featuresTracked] = trackFeatures(gray);
    
    // Estimate motion between frames
    cv::Mat H_prev2curr = estimateMotion(currPoints, featuresTracked);
    
    // Get current frame index
    uint64_t current_idx = stabilizationWindow_.frames.back().frame_idx;
    
    // Update transformations window
    updateTransformations(H_prev2curr, current_idx);
    
    // Determine which frame to present
    size_t presentation_frame_idx = 0;
    if (stabilizationWindow_.frames.size() > totalFutureFrames_) {
        presentation_frame_idx = stabilizationWindow_.frames.size() - totalFutureFrames_ - 1;
    }

    cv::Mat H_stabilize;

    const double epsilon = 1e-6;
    
    cv::Mat H_global_smoothing = calculateGlobalSmoothingStabilization(presentation_frame_idx);
    cv::Mat H_lock = calculateFullLockStabilization(presentation_frame_idx);
    
    cv::Point2d rot_center = cv::Point2d(workingSize_.width / 2, workingSize_.height / 2);

    HomographyParameters homography_params_global_smoothing = decomposeHomography(H_global_smoothing, rot_center);
    cv::Mat H_global_smoothing2 = composeHomography(homography_params_global_smoothing);
    assert(cv::norm(H_global_smoothing - H_global_smoothing2, cv::NORM_INF) < epsilon);
    
    HomographyParameters homography_params_lock = decomposeHomography(H_lock, rot_center);
    cv::Mat H_lock2 = composeHomography(homography_params_lock);
    assert(cv::norm(H_lock - H_lock2, cv::NORM_INF) < epsilon);

    HomographyParameters homography_params_stabilize;
    cv::Vec2d t_shifted;

    // Calculate stabilization transform based on selected mode
    switch (stabilizationMode_) {
        case StabilizationMode::FULL_LOCK:
            H_stabilize = H_lock; // simply use the full lock transform
            std::cout <<" Homography lock theta: " << homography_params_lock.theta * 180.0 / M_PI << " deg" << std::endl;
            std::cout <<" Homography lock t: " << homography_params_lock.t << std::endl;
            std::cout <<" Homography lock s: " << homography_params_lock.s << std::endl;

            break;
        case StabilizationMode::TRANSLATION_LOCK:
            // Currently handled same as GLOBAL_SMOOTHING
            // TODO: Implement specific translation-only stabilization
            homography_params_stabilize = homography_params_global_smoothing;
            std::cout << "prior homography_params_lock.t: " << homography_params_lock.t << std::endl;

            t_shifted = homography_params_lock.t;
            t_shifted = removeRotationCenterFromTranslation(homography_params_lock.t, homography_params_lock.c, homography_params_lock.s, homography_params_lock.theta);
            std::cout << "post homography_params_lock.t: " << t_shifted << std::endl;

            homography_params_stabilize.t = t_shifted;
            H_stabilize = composeHomography(homography_params_stabilize);
            break;
        case StabilizationMode::ROTATION_LOCK:
            // Currently handled same as GLOBAL_SMOOTHING
            // TODO: Implement specific rotation-only stabilization
            H_stabilize = calculateGlobalSmoothingStabilization(presentation_frame_idx);
            break;
        case StabilizationMode::GLOBAL_SMOOTHING:
        default:
            H_stabilize = H_global_smoothing;
            break;
    }
    
    // Scale stabilization transform to match original resolution
    cv::Mat H_stabilize_scaled = H_stabilize.clone();
    if (scaleFactor_ != 1.0) {
        // Adjust translation components
        H_stabilize_scaled.at<double>(0, 2) /= scaleFactor_;
        H_stabilize_scaled.at<double>(1, 2) /= scaleFactor_;
    }
    
    // Warp frame for stabilization
    cv::Mat stabilized = warpFrame(frame, H_stabilize_scaled, next_trail_background, presentation_frame_idx);
    
    // Detect new features for next frame
    detectNewFeatures(gray);
    
    // Print timing information periodically
    printTimings();
    
    // Update trail background for next frame
    trail_background_ = stabilized.clone();
    
    return stabilized;
}

/**
 * @brief Computes the QR decomposition of a 2x2 matrix using the Gram-Schmidt process.
 *
 * Given a 2x2 matrix A, computes matrices Q (orthogonal) and R (upper triangular)
 * such that A = QR.
 *
 * @param A Input 2x2 matrix (must be of type CV_64F).
 * @param Q Output 2x2 orthogonal matrix (CV_64F).
 * @param R Output 2x2 upper triangular matrix (CV_64F).
 * @throws std::invalid_argument if A is not a 2x2 CV_64F matrix.
 * @throws std::runtime_error if the columns of A are linearly dependent (cannot normalize).
 */
void qrDecomposition2x2(const cv::Mat& A, cv::Mat& Q, cv::Mat& R) {
    // --- Input Validation ---
    if (A.rows != 2 || A.cols != 2) {
        throw std::invalid_argument("Input matrix A must be 2x2.");
    }
    if (A.type() != CV_64F) {
        throw std::invalid_argument("Input matrix A must be of type CV_64F (double precision).");
    }

    if (std::abs(cv::determinant(A)) < 1e-6) {
        throw std::invalid_argument("Input matrix A is singular. QR decomposition requires non-singular matrix.");
    }

    // --- Initialization ---
    // Create output matrices with the correct size and type
    Q = cv::Mat::zeros(2, 2, CV_64F);
    R = cv::Mat::zeros(2, 2, CV_64F);

    // Get column vectors (creating copies)
    cv::Mat a1 = A.col(0);
    cv::Mat a2 = A.col(1);

    // --- Gram-Schmidt Process ---

    const double epsilon = 1e-6;

    // Process first column (a1)
    double norm_a1 = cv::norm(a1);
    if (norm_a1 < epsilon) { // Check for zero vector
        throw std::runtime_error("First column is zero or near-zero. QR decomposition requires linearly independent columns.");
    }
    cv::Mat q1 = a1 / norm_a1;

    // Process second column (a2)
    double r12 = a2.dot(q1); // Projection coefficient: R[0, 1] = q1^T * a2
    cv::Mat u2 = a2 - r12 * q1; // Orthogonal vector
    double norm_u2 = cv::norm(u2);

    if (norm_u2 < epsilon) { // Check for linear dependence (a2 is parallel to a1)
        throw std::runtime_error("Columns are linearly dependent. QR decomposition requires linearly independent columns.");
    }
    cv::Mat q2 = u2 / norm_u2;

    // --- Construct Q and R Matrices ---

    // Fill Q matrix with the orthonormal vectors q1 and q2
    q1.copyTo(Q.col(0));
    q2.copyTo(Q.col(1));

    // Fill R matrix (upper triangular)
    R.at<double>(0, 0) = norm_a1; // R[0, 0] = ||a1||
    R.at<double>(0, 1) = r12;     // R[0, 1] = q1.dot(a2)
    R.at<double>(1, 0) = 0.0;     // R[1, 0] = 0
    R.at<double>(1, 1) = norm_u2; // R[1, 1] = ||u2||

    // Assert that QR = A
    double max_diff_A = cv::norm(A - Q*R, cv::NORM_INF);
    if (max_diff_A > epsilon) {
        throw std::runtime_error("QR decomposition failed. Max difference: " + std::to_string(max_diff_A));
    }

    // Assert that Q is orthogonal
    cv::Mat Q_test = Q.t() * Q;
    cv::Mat I = cv::Mat::eye(2, 2, CV_64F);
    double max_diff_Q = cv::norm(Q_test - I, cv::NORM_INF);
    if (max_diff_Q > epsilon) {
        throw std::runtime_error("Q is not orthogonal. Max difference: " + std::to_string(max_diff_Q));
    }

    // Assert that R is upper triangular
    R.at<double>(1, 0) = 0.0; // R[1, 0] should be 0
}


HomographyParameters Stabilizer::decomposeHomography(const cv::Mat& H, cv::Point2d rot_center) {
    if (H.empty() || H.rows != 3 || H.cols != 3 || H.type() != CV_64F) {
        throw std::invalid_argument("Error: Input homography matrix must be a non-empty 3x3 CV_64F matrix.");
    }

    const double epsilon = 1e-6;

    // Normalize H such that H(2, 2) = 1
    double h33 = H.at<double>(2, 2);
    if (std::abs(h33) < epsilon) {
        throw std::invalid_argument("Error: H(2, 2) is close to zero. Degenerate homography.");
    }
    cv::Mat H_norm = H / h33;

    // Extract translation vector t = [tx, ty] and projective vector v = [v1, v2]
    cv::Mat t = H_norm(cv::Rect(2, 0, 1, 2));
    cv::Mat v = H_norm(cv::Rect(0, 2, 2, 1));

    // Extract A matrix from H_norm
    cv::Mat A = H_norm(cv::Rect(0, 0, 2, 2));

    // Extract RK matrix and scaling factor s from A
    cv::Mat sRK = A - t * v;

    double det_sRK = cv::determinant(sRK);
    if (std::abs(det_sRK) < epsilon) {
        throw std::invalid_argument("Error: determinant of sRK is close to zero. Degenerate homography.");
    }
    double s = std::sqrt(det_sRK); // sqrt because A is 2x2 matrix
    cv::Mat RK = sRK / s;

    // Extrac R and K from RK using QR-decomposition
    // They are both 2x2 matrices, and there is a closed-form solution for them
    // R = [cos(theta) -sin(theta); sin(theta) cos(theta)], det(R) = 1, R is a rotation matrix, R^T = R^-1
    // K = [k1 d; 0 k2], k1*k2 = det(K) = 1, K is a upper triangular matrix
    cv::Mat R, K;
    qrDecomposition2x2(RK, R, K);

    // Extract angle theta from R
    double cos_theta = (R.at<double>(0, 0) + R.at<double>(1, 1)) / 2;
    double sin_theta = (R.at<double>(1, 0) - R.at<double>(0, 1)) / 2;
    double theta = std::atan2(sin_theta, cos_theta);
    
    // Extract k and delta from K
    double k1 = K.at<double>(0, 0);
    double k2 = K.at<double>(1, 1);
    assert(std::abs(k2 - 1/k1) < epsilon); // K is a upper triangular matrix with det(K) = 1, hence k2 = 1/k1

    double delta = K.at<double>(0, 1);

    // Now we need to shift translation t given scaling s and rotation center rot_center
    cv::Mat I = cv::Mat::eye(2, 2, CV_64F);
    cv::Vec2d t_shift = s * cv::Vec2d(cv::Mat((I - R) * cv::Vec2d(rot_center)));
    cv::Vec2d t_shifted = cv::Vec2d(t) - t_shift;

    // only need k1, as k2 = 1/k1
    return HomographyParameters{s, theta, rot_center, k1, delta, t_shifted, v};
}

cv::Mat Stabilizer::composeHomography(const HomographyParameters& params) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat R = (cv::Mat_<double>(2, 2) << 
                 std::cos(params.theta), -std::sin(params.theta),
                 std::sin(params.theta), std::cos(params.theta));
    
    cv::Mat K = (cv::Mat_<double>(2, 2) << 
                 params.k, params.delta,
                 0, 1/params.k);

    // Need to shift translation vector t given scaling s and rotation center
    cv::Mat I = cv::Mat::eye(2, 2, CV_64F);

    // t_shift = s * (I - R) * rot_center
    cv::Vec2d t_shift = params.s * cv::Vec2d(cv::Mat((I - R) * cv::Vec2d(params.c)));
    cv::Vec2d t_shifted = params.t + t_shift;

    cv::Mat A = params.s * R * K + t_shifted * params.v.t();
    cv::Mat H_norm = cv::Mat::eye(3, 3, CV_64F);

    A.copyTo(H_norm(cv::Rect(0, 0, 2, 2)));

    H_norm.at<double>(0, 2) = t_shifted[0];
    H_norm.at<double>(1, 2) = t_shifted[1];

    H_norm.at<double>(2, 0) = params.v[0];
    H_norm.at<double>(2, 1) = params.v[1];

    return H_norm;
}
