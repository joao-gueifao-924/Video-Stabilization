#include "stabilizer.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> // For cvtColor
#include <opencv2/video/tracking.hpp> // Need this for LK
#include <vector>
#include <cmath>
#include <stdexcept>
#include <deque>   // Include deque
#include <iostream> // For potential debug/error messages
#include <chrono>   // Include chrono for timing


const int MIN_POINTS_FOR_MOTION_ESTIMATION = 10;
const bool REFINE_WITH_ECC = false;

Stabilizer::Stabilizer(size_t pastFrames, size_t futureFrames, int workingHeight)
    : totalPastFrames_(pastFrames), totalFutureFrames_(futureFrames),
      workingHeight_(workingHeight)
{
    if (pastFrames == 0 && futureFrames == 0) 
        throw std::invalid_argument("Stabilizer: pastFrames and futureFrames cannot both be 0");

    const double min_working_height = 90.0; // This is the minimum working height for the stabilizer to work properly
    if (workingHeight <= min_working_height)
        throw std::invalid_argument("Stabilizer: workingHeight must be greater than " + std::to_string(min_working_height));
    if (workingHeight > 2160) // just some sobe big arbitrary number but still sensible
        throw std::invalid_argument("Stabilizer: workingHeight must be no more than 2160");

    reset();
    last_print_time_ = now();
}

void Stabilizer::reset() {
    prevGray_ = cv::Mat();
    prevPoints_.clear();
    trail_background_ = cv::Mat();
    stabilizationWindow_.transformations.clear();
    stabilizationWindow_.frames.clear();
    original_frame_size_ = cv::Size(0, 0);
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
    
    // Reset ORB- and SIFT-based motion estimation variables
    referenceFrameIdx_ = -1;
    referenceGray_ = cv::Mat();
    orb_referenceKeypoints_.clear();
    orb_referenceDescriptors_ = cv::Mat();
    sift_referenceKeypoints_.clear();
    sift_referenceDescriptors_ = cv::Mat();

    accumulatedTransform_ = Transformation();
    stabilizationMode_ = StabilizationMode::GLOBAL_SMOOTHING;

    // Reset feature detectors
    orb_.reset();
    sift_.reset();
}

void Stabilizer::setStabilizationMode(StabilizationMode mode) {
    if (stabilizationMode_ != mode) {

        // Reset variables used for ORB- and SIFT-based motion estimation when switching modes
        referenceFrameIdx_ = -1; // Reset reference frame
        referenceGray_ = cv::Mat();
        orb_referenceKeypoints_.clear();
        orb_referenceDescriptors_ = cv::Mat();
        sift_referenceKeypoints_.clear();
        sift_referenceDescriptors_ = cv::Mat();

        accumulatedTransform_ = Transformation();

        stabilizationMode_ = mode;
        std::cout << "Stabilization mode changed to: ";
        switch (mode) {
            case StabilizationMode::ACCUMULATED_FULL_LOCK:
                std::cout << "ACCUMULATED_FULL_LOCK";
                break;
            case StabilizationMode::ORB_FULL_LOCK:
                std::cout << "ORB_FULL_LOCK";
                break;
            case StabilizationMode::SIFT_FULL_LOCK:
                std::cout << "SIFT_FULL_LOCK";
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

        // Reset feature detectors
        orb_.reset();
        sift_.reset();
    }
}

void Stabilizer::initializeFrame(const cv::Mat& frame) {
    if (frame.rows <= 10 || frame.cols <= 10) {
        throw std::invalid_argument("Stabilizer: Frame has invalid size. Rows: " + std::to_string(frame.rows) + ", Cols: " + std::to_string(frame.cols));
    }

    // Store original size if first frame. Error out if size has changed.
    const bool frame_size_is_already_set = original_frame_size_.width > 0 && original_frame_size_.height > 0;
    const bool frame_size_has_changed = original_frame_size_.width != frame.cols || original_frame_size_.height != frame.rows;
    
    if (frame_size_has_changed) {
        if (frame_size_is_already_set) { 
            throw std::invalid_argument("Stabilizer: Frame size has changed. This is not supported.");
        }
        original_frame_size_ = frame.size();
        // Calculate working size maintaining aspect ratio
        scaleFactor_ = static_cast<double>(workingHeight_) / frame.rows;
        workingSize_ = cv::Size(static_cast<int>(frame.cols * scaleFactor_), workingHeight_);
    }
    
    // Initialize or resize trail background
    if (trail_background_.empty() || trail_background_.size() != original_frame_size_) {
        trail_background_ = cv::Mat::zeros(original_frame_size_, frame.type());
    }
}

cv::Mat Stabilizer::prepareTrailBackground() {
    cv::Mat next_trail_background = trail_background_.clone();
    // Grayscale
    //cv::cvtColor(trail_background_, next_trail_background, cv::COLOR_BGR2GRAY);
    // Convert back to BGR for compatibility
    //cv::cvtColor(next_trail_background, next_trail_background, cv::COLOR_GRAY2BGR);

    const double image_ratio = static_cast<double>(trail_background_.cols) / static_cast<double>(trail_background_.rows);
    const int downscaled_rows = 100;
    const int downscaled_cols = static_cast<int>(downscaled_rows * image_ratio);

    // cv::resize(trail_background_, next_trail_background, cv::Size(downscaled_cols, downscaled_rows), 0, 0, cv::INTER_NEAREST);
    // cv::GaussianBlur(next_trail_background, next_trail_background, cv::Size(5, 5), 0);
    // cv::resize(next_trail_background, next_trail_background, trail_background_.size(), 0, 0, cv::INTER_LINEAR);
    // cv::GaussianBlur(next_trail_background, next_trail_background, cv::Size(7, 7), 0);

    // Darken
    //next_trail_background *= 0.99; 
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

void Stabilizer::trackFeatures(const cv::Mat& previous_gray, const cv::Mat& current_gray, 
                                const std::vector<cv::Point2f> & previousPoints,
                                std::vector<cv::Point2f> & filtered_previousPoints,
                                std::vector<cv::Point2f> & filtered_currentPoints) {
    filtered_previousPoints.clear();
    filtered_currentPoints.clear();

    if (previousPoints.empty()) return;

    std::vector<cv::Point2f> currPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    
    auto start_lk = now();
    const cv::Size WINDOW_SIZE = cv::Size(21, 21);
    const int MAX_PYRAMID_LEVEL = 3;
    const cv::TermCriteria TERM_CRITERIA(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.01);
    const int FLAGS = 0;
    const double MIN_EIGENVAL = 1e-4;

    cv::calcOpticalFlowPyrLK(previous_gray, current_gray, previousPoints, currPoints, status, err,
                            WINDOW_SIZE, MAX_PYRAMID_LEVEL, TERM_CRITERIA, FLAGS, MIN_EIGENVAL);
    auto end_lk = now();
    auto duration_ms_lk = toMilliseconds(start_lk, end_lk);
    lk_call_count_++;
    lk_avg_duration_ms_ += (duration_ms_lk - lk_avg_duration_ms_) / lk_call_count_;

    // Filter out points that were not successfully tracked

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] == 1) { // Check if the point was tracked successfully
            filtered_previousPoints.push_back(previousPoints[i]);
            filtered_currentPoints.push_back(currPoints[i]);
        }
    }
}

cv::Mat Stabilizer::estimateMotion(const std::vector<cv::Point2f>& previousPoints, const std::vector<cv::Point2f>& currentPoints) {
    cv::Mat H_prev2curr = cv::Mat::eye(3, 3, CV_64F);

    if (currentPoints.size() < MIN_POINTS_FOR_MOTION_ESTIMATION) return H_prev2curr; // identity matrix response
    
    auto start_motion_estimation = now();

    #if 1 // partial 2D Euclidean motion estimation (isotropic scaling, in-image-plane rotation, in-image-plane translation)
        cv::Mat M = cv::estimateAffinePartial2D(previousPoints, currentPoints, cv::noArray(), cv::RANSAC);
    #else // full Homography estimation (isotropic scaling, in-image-plane rotation, in-image-plane translation, anisotropic scaling, shear, line at infinity shift)
        cv::Mat M = cv::findHomography(previousPoints, currentPoints, cv::RANSAC);
    #endif
    auto end_motion_estimation = now();


    auto duration_ms_motion = toMilliseconds(start_motion_estimation, end_motion_estimation);
            
    homography_call_count_++;
    homography_avg_duration_ms_ += 
        (duration_ms_motion - homography_avg_duration_ms_) / homography_call_count_;

    if (M.empty() || !cv::checkRange(M)) return H_prev2curr; // // identity matrix response

    if (M.rows == 2 && M.cols == 3) {
        // Convert 2x3 matrix M = [sR | t] to 3x3 homography H
        H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
        M.copyTo(H_prev2curr(cv::Rect(0, 0, 3, 2)));

    } else { // M is already a 3x3 homography
        M.copyTo(H_prev2curr);
    }

    // Let's kill isotropic scaling from the estimated motion (it is typically unstable) - this enhances visual stability.
    // In the future, we could implement a estimateRigidBodyMotion() that would fit in-image-plane rotation and translation
    // with fixed unitary scaling factor, so that error is minimized.
    // Define center of scaling (and rotation) as the center of the image. If we don't use this, we
    // we will see weird scaling artifacts coming from the top left corner of the image when we kill scaling.
    cv::Point2d rot_center(workingSize_.width / 2.0, workingSize_.height / 2.0);
    HomographyParameters homography_params;
    if(decomposeHomography(H_prev2curr, homography_params, rot_center))
    {
        homography_params.s = 1.0;
        H_prev2curr = composeHomography(homography_params, rot_center);
    }
    else
    {
        std::cerr << "Error: Failed to decompose homography" << std::endl;
        H_prev2curr = cv::Mat::eye(3, 3, CV_64F);
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

std::pair<std::vector<cv::KeyPoint>, cv::Mat> filterKeypointByRelativeSize(int image_height, std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors, float max_relative_size = 0.05f) {
    // Filter keypoints by size: discard if too big compared to image height:
    // Filter out keypoints that are too large compared to image height
    const float maxAllowedSize = image_height * max_relative_size;
    
    std::vector<cv::KeyPoint> filtered_keypoints;
    cv::Mat filtered_descriptors;
    
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].size < maxAllowedSize) {
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row(i));
        }
    }
    return {filtered_keypoints, filtered_descriptors};
}

cv::Mat Stabilizer::calculateFullLockStabilization(size_t presentation_frame_idx) {
    // If we are not using any lock, we can just return the identity matrix
    if (stabilizationMode_ == StabilizationMode::GLOBAL_SMOOTHING) {
        return cv::Mat::eye(3, 3, CV_64F);
    }

    if (stabilizationMode_ == StabilizationMode::ACCUMULATED_FULL_LOCK) 
    {
        size_t frame_idx = stabilizationWindow_.frames[presentation_frame_idx].frame_idx;

        // If this iteration is the first time we are computing the stabilization transform, use the identity matrix
        if (accumulatedTransform_.H.empty()) {
            accumulatedTransform_.H = cv::Mat::eye(3, 3, CV_64F);
            accumulatedTransform_.from_frame_idx = frame_idx;
            accumulatedTransform_.to_frame_idx = frame_idx;
        }
        else {
            assert(presentation_frame_idx > 0);
            Transformation next_transform = stabilizationWindow_.transformations[presentation_frame_idx -1];
            assert(next_transform.from_frame_idx == accumulatedTransform_.to_frame_idx);

            // update the accumulated transform by multiplying it with the transformation
            accumulatedTransform_.H = accumulatedTransform_.H * next_transform.H;
            accumulatedTransform_.to_frame_idx = next_transform.to_frame_idx;
        }

        #if 0
            // We now have the full accumulated transformation from the anchor frame to the presentation frame
            // Let's decompose it to get rotation angle and translation vector
            HomographyParameters homography_params_lock;
            
            // Ensure workingSize_ is valid before calculating image_center
            if (workingSize_.width <= 0 || workingSize_.height <= 0) {
                throw std::runtime_error("Stabilizer::calculateFullLockStabilization: workingSize_ is invalid. Width: " 
                                        + std::to_string(workingSize_.width) + ", Height: " + std::to_string(workingSize_.height));
            }
            const cv::Point2d image_center(workingSize_.width / 2.0, workingSize_.height / 2.0);

            // Check if accumulatedTransform_.H is valid before decomposition
            if (accumulatedTransform_.H.size() != cv::Size(3,3) || !cv::checkRange(accumulatedTransform_.H)) {
                 throw std::runtime_error("Warning: accumulatedTransform_.H is empty or contains NaN/Inf before decomposition. Frame idx: " + std::to_string(frame_idx));
            }

            if (!decomposeHomography(accumulatedTransform_.H, homography_params_lock, image_center)) {
                throw std::runtime_error("Warning: decomposeHomography failed for frame_idx: " + std::to_string(frame_idx));
            }

            // Check homography_params_lock.t for NaN/Inf as an extra precaution, though decomposeHomography should ideally handle it
            if (!cv::checkRange(cv::Mat(homography_params_lock.t))) { // Convert Vec2d to Mat for checkRange
                throw std::runtime_error("Warning: homography_params_lock.t contains NaN/Inf after successful decomposition. Frame idx: " + std::to_string(frame_idx));
            }
            // Also check other critical params from decomposition
            if (!std::isfinite(homography_params_lock.s) || !std::isfinite(homography_params_lock.theta)) {
                throw std::runtime_error("Warning: homography_params_lock.s or .theta is NaN/Inf. Frame idx: " + std::to_string(frame_idx));
            }

            double accumulated_rotation_angle = homography_params_lock.theta;
            cv::Vec2d accumulated_translation = homography_params_lock.t;

            std::cout << "accumulated_rotation_angle: " << accumulated_rotation_angle * 180.0 / M_PI << " deg" << std::endl;
            std::cout << "accumulated_translation: " << accumulated_translation(0) << ", " << accumulated_translation(1) << std::endl;

            
            // The following achieves rotation lock successfully!! Keep it!
            HomographyParameters new_params = homography_params_lock;
            new_params.theta = -homography_params_lock.theta;
            new_params.t = cv::Vec2d(0,0);
            cv::Mat H_rotation_lock = composeHomography(new_params, cv::Vec2d(0,0));

            // To achieve full lock, we first need to untranslate the frame by the accumulated translation vector
            // and only then de-rotate the frame by the accumulated rotation angle.
            cv::Mat H_translation_lock = cv::Mat::eye(3, 3, CV_64F);
            H_translation_lock.at<double>(0, 2) = -accumulated_translation(0);
            H_translation_lock.at<double>(1, 2) = -accumulated_translation(1);

            HomographyParameters derotate_params;
            // Set the desired de-rotation angle
            if (!std::isfinite(accumulated_rotation_angle)) {
                 throw std::runtime_error("Critical: accumulated_rotation_angle is NaN/Inf before creating derotate_params. Frame idx: " + std::to_string(frame_idx));
            }
            derotate_params.theta = -accumulated_rotation_angle;
            
            cv::Mat H_derotate = composeHomography(derotate_params, image_center);
            if (!cv::checkRange(H_derotate)) {
                 throw std::runtime_error("Critical: H_derotate matrix contains NaN/Inf. Frame idx: " + std::to_string(frame_idx));
            }
            
            cv::Mat H_lock = H_translation_lock * H_derotate;



            // HomographyParameters new_params = homography_params_lock;
            // new_params.theta = 0.0;
            // new_params.t = -homography_params_lock.t;
            // cv::Mat H_lock = composeHomography(new_params, cv::Vec2d(0,0));

            // Use the accumulated transform to stabilize the frame
            return H_lock;

        #endif

        return accumulatedTransform_.H.inv();
    }
    else if (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK || stabilizationMode_ == StabilizationMode::SIFT_FULL_LOCK) { // direct reference frame to current frame motion estimation
        // Get the current frame from the presentation index
        cv::Mat currentFrame = stabilizationWindow_.frames[presentation_frame_idx].image;

        static cv::Mat previouslyReturnedH = cv::Mat::eye(3, 3, CV_64F);
        
        // Downscale current frame to working resolution
        cv::Mat currentResized;
        cv::resize(currentFrame, currentResized, workingSize_, 0, 0, cv::INTER_NEAREST);
        
        // Convert to grayscale
        cv::Mat currentGray;
        cv::cvtColor(currentResized, currentGray, cv::COLOR_BGR2GRAY);

        // We need to improve the greyscale image for feature detection.
        // We do this by applying a series of filters in the following order:
        //   1. Median blur (to remove noise)
        //   2. Sharpening (to enhance edges)
        //   3. CLAHE (to enhance contrast)
        //   4. Median blur (again)

        cv::medianBlur(currentGray, currentGray, 5);

        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
                            0, -1,  0,
                            -1,  5, -1,
                            0, -1,  0); // A common sharpening kernel
        cv::filter2D(currentGray, currentGray, -1, kernel);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(2.0); // Adjust clip limit (e.g., 2.0-4.0)
        clahe->setTilesGridSize(cv::Size(8, 8)); // Adjust grid size
        clahe->apply(currentGray, currentGray);

        cv::medianBlur(currentGray, currentGray, 5);
        
        // Find matched point pairs for homography calculation
        std::vector<cv::Point2f> refPoints, currPoints;

        // ORB parameters (defined here for clarity, used if mode is ORB_FULL_LOCK)
        const int ORB_MAX_FEATURES = 2500;        // The maximum number of features to retain
        const float ORB_SCALE_FACTOR = 1.2f;      // Pyramid decimation ratio, greater than 1
        const int ORB_PYRAMID_LEVELS = 12;        // The number of pyramid levels
        const int ORB_EDGE_THRESHOLD = 31;        // Size of the border where features are not detected
        const int ORB_FIRST_LEVEL = 0;            // The level of pyramid to put source image at
        const int ORB_WTA_K = 2;                  // The number of points that produce each element of the oriented BRIEF descriptor
        const cv::ORB::ScoreType ORB_SCORE_TYPE = cv::ORB::FAST_SCORE;  // The default HARRIS_SCORE=0, FAST_SCORE=1
        const int ORB_PATCH_SIZE = 31;            // Size of the patch used by the oriented BRIEF descriptor
        const int ORB_FAST_THRESHOLD = 20;        // The FAST threshold for feature detection
        const float LOWE_RATIO_THRESH = 0.6f;     // Threshold for Lowe's ratio test in feature matching
        const float MAX_ORB_KEYPOINT_SIZE_RATIO = 0.10f;  // Maximum allowed keypoint size as ratio of image height

        // SIFT parameters (defined here for clarity, used if mode is SIFT_FULL_LOCK)
        const int SIFT_N_FEATURES = 2500;      // number of best features to retain (0 = unlimited)
        const int SIFT_N_OCTAVE_LAYERS = 3;    // number of layers in each octave
        const double SIFT_CONTRAST_THRESHOLD = 0.04; // filter out weak features in semi-uniform (low-contrast) regions (higher values yield less features)
        const double SIFT_EDGE_THRESHOLD = 5;  // filter out edge-like features versus corner-like features (higher values yield more features)
        const double SIFT_SIGMA = 1.2;         // sigma of Gaussian applied to first octave
        const float SIFT_MAX_KEYPOINT_SIZE_RATIO = 0.05f; // Maximum allowed keypoint size as ratio of image height

        // RANSAC parameters (common for both ORB and SIFT motion estimation)
        const double MAX_RANSAC_REPROJ_THRESHOLD = 5.0;
        const int ROBUST_METHOD_MOTION_ESTIMATION = cv::RANSAC;

        // If we don't have a reference/anchor image yet (referenceGray_ is still empty), do the following:
        //   1. Set this frame as reference
        //   2. Initialize the feature detector
        //   3. Filter the keypoints by size
        //   4. Return the identity matrix (no motion to cancel out)

        if (referenceGray_.empty()) {
            std::cout << "Initializing ORB/SIFT FULL_LOCK with new reference image" << std::endl;
            referenceFrameIdx_ = stabilizationWindow_.frames[presentation_frame_idx].frame_idx;
            currentGray.copyTo(referenceGray_); // This referenceGray_ is shared for ORB & SIFT
            previouslyReturnedH = cv::Mat::eye(3, 3, CV_64F); // Reset fallback homography

            std::vector<cv::KeyPoint> filtered_keypoints_orb;
            cv::Mat filtered_descriptors_orb;
            std::vector<cv::KeyPoint> filtered_keypoints_sift;
            cv::Mat filtered_descriptors_sift;

            if (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK)
            {
                orb_ = cv::ORB::create(ORB_MAX_FEATURES,
                                    ORB_SCALE_FACTOR,
                                    ORB_PYRAMID_LEVELS,
                                    ORB_EDGE_THRESHOLD,
                                    ORB_FIRST_LEVEL, 
                                    ORB_WTA_K,
                                    ORB_SCORE_TYPE,
                                    ORB_PATCH_SIZE,
                                    ORB_FAST_THRESHOLD);
            }
            else if (stabilizationMode_ == StabilizationMode::SIFT_FULL_LOCK)
            {
                sift_ = cv::SIFT::create(SIFT_N_FEATURES, 
                                        SIFT_N_OCTAVE_LAYERS, 
                                        SIFT_CONTRAST_THRESHOLD, 
                                        SIFT_EDGE_THRESHOLD, 
                                        SIFT_SIGMA);
            }

            if (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK) {
                orb_->detectAndCompute(referenceGray_, cv::noArray(), 
                                    orb_referenceKeypoints_, 
                                    orb_referenceDescriptors_);
                
                auto [filtered_kpts, filtered_desc] = filterKeypointByRelativeSize(
                                                                            referenceGray_.rows,
                                                                            orb_referenceKeypoints_,
                                                                            orb_referenceDescriptors_,
                                                                            MAX_ORB_KEYPOINT_SIZE_RATIO
                );
                
                orb_referenceKeypoints_ = filtered_kpts;
                orb_referenceDescriptors_ = filtered_desc;

            } else { // SIFT_FULL_LOCK
                sift_->detectAndCompute(referenceGray_, cv::noArray(),
                                     sift_referenceKeypoints_,
                                     sift_referenceDescriptors_);
                
                auto [filtered_kpts, filtered_desc] = filterKeypointByRelativeSize(
                                                                            referenceGray_.rows,
                                                                            sift_referenceKeypoints_, 
                                                                            sift_referenceDescriptors_,
                                                                            SIFT_MAX_KEYPOINT_SIZE_RATIO
                );
                
                sift_referenceKeypoints_ = filtered_kpts;
                sift_referenceDescriptors_ = filtered_desc;
            }
            
            return cv::Mat::eye(3, 3, CV_64F); // First frame is reference frame, so there is no motion to cancel out.
        }
        
        // For subsequent frames:
        //   1. Detect features on incoming frame.
        //   2. Match features to reference and filter matches. 
        //   3. If not enough matches, return the identity matrix.
        //   4. If enough matches, compute homography between reference and current frame. If not, return the identity matrix.
        
        try {
            std::vector<cv::KeyPoint> currentKeypoints;
            cv::Mat currentDescriptors;
            std::vector<cv::KeyPoint> filtered_keypoints_curr;
            cv::Mat filtered_descriptors_curr;

            if (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK) {
                orb_->detectAndCompute(currentGray, cv::noArray(), 
                                      currentKeypoints, currentDescriptors);

                std::tie(filtered_keypoints_curr, filtered_descriptors_curr) = filterKeypointByRelativeSize(
                                                                            currentGray.rows, 
                                                                            currentKeypoints, 
                                                                            currentDescriptors, 
                                                                            MAX_ORB_KEYPOINT_SIZE_RATIO);
            } else { // SIFT_FULL_LOCK
                sift_->detectAndCompute(currentGray, cv::noArray(), 
                                       currentKeypoints, currentDescriptors);

                std::tie(filtered_keypoints_curr, filtered_descriptors_curr) = filterKeypointByRelativeSize(
                                                                             currentGray.rows,
                                                                             currentKeypoints, 
                                                                             currentDescriptors, 
                                                                             SIFT_MAX_KEYPOINT_SIZE_RATIO);
            }

            currentKeypoints  = filtered_keypoints_curr;
            currentDescriptors = filtered_descriptors_curr;
            
            // display feature points for debugging - KEEP THIS!
            #if 1
                cv::Mat img_debug = currentGray.clone();
                cv::drawKeypoints(img_debug, currentKeypoints, img_debug, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::imshow("features", img_debug);
            #endif

            // If not enough features, can't stabilize
            const auto& referenceKeypoints = (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK) ? orb_referenceKeypoints_ : sift_referenceKeypoints_;
            if (currentKeypoints.size() < 10 || referenceKeypoints.size() < 10) {
                std::cout << "Not enough features for ORB/SIFT FULL_LOCK" << std::endl;
                return previouslyReturnedH; 
            }

            // Match features
            std::vector<cv::DMatch> good_matches;
            if (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK) {
                cv::BFMatcher matcher(cv::NORM_HAMMING);
                std::vector<std::vector<cv::DMatch>> knn_matches;
                // It's generally better to use knnMatch and apply Lowe's ratio test for better quality matches
                // k=2 means find the two best matches for each descriptor
                matcher.knnMatch(orb_referenceDescriptors_, currentDescriptors, knn_matches, 2);
                
                // Filter matches using Lowe's ratio test
                for (size_t i = 0; i < knn_matches.size(); i++) {
                    if (knn_matches[i].size() == 2) { // Ensure we have two matches
                        if (knn_matches[i][0].distance < LOWE_RATIO_THRESH * knn_matches[i][1].distance) {
                            good_matches.push_back(knn_matches[i][0]);
                        }
                    }
                }
                std::cout << "Good ORB matches after ratio test: " << good_matches.size() << std::endl;
                 if (good_matches.size() < MIN_POINTS_FOR_MOTION_ESTIMATION) {
                     std::cout << "Not enough good ORB matches for FULL_LOCK" << std::endl;
                     return previouslyReturnedH;
                 }

            } else { // SIFT_FULL_LOCK
                cv::FlannBasedMatcher matcher; // SIFT typically uses FlannBasedMatcher
                std::vector<cv::DMatch> matches;
                matcher.match(sift_referenceDescriptors_, currentDescriptors, matches);

                // Compute average distance among matches[i]
                double avg_dist = 0;
                for (int i = 0; i < sift_referenceDescriptors_.rows; i++) {
                    avg_dist += matches[i].distance;
                }
                avg_dist /= sift_referenceDescriptors_.rows;

                // Store good matches based on distance
                for (int i = 0; i < sift_referenceDescriptors_.rows; i++) {
                    if (matches[i].distance <= std::max(avg_dist * 0.5, 0.02)) { // Adjusted threshold
                        good_matches.push_back(matches[i]);
                    }
                }
                int pct_good_matches = matches.empty() ? 0 : round(100.0 * (double)good_matches.size() / (double)matches.size());
                std::cout << "Total good SIFT matches: " << good_matches.size() << ", " << pct_good_matches << "%" << std::endl;
                 if (good_matches.size() < MIN_POINTS_FOR_MOTION_ESTIMATION) {
                    std::cout << "Not enough good SIFT matches for FULL_LOCK" << std::endl;
                    return previouslyReturnedH;
                }
            }
       
            // Extract location of good matches
            const auto& refKps = (stabilizationMode_ == StabilizationMode::ORB_FULL_LOCK) ? orb_referenceKeypoints_ : sift_referenceKeypoints_;
            for (size_t i = 0; i < good_matches.size(); i++) {
                refPoints.push_back(refKps[good_matches[i].queryIdx].pt);
                currPoints.push_back(currentKeypoints[good_matches[i].trainIdx].pt);
            }

        }
        catch (const std::exception& e) {
            std::cerr << "Error in FULL_LOCK calculation: " << e.what() << std::endl;
            return previouslyReturnedH;
        }
        
        // Compute homography from reference to current (direct mapping)
        cv::Mat H_prev2current = cv::Mat::eye(3, 3, CV_64F);

        if (refPoints.size() < MIN_POINTS_FOR_MOTION_ESTIMATION)
        {
            std::cerr << "Error: not enough keypoint matches to estimate motion: " << std::endl;
            return previouslyReturnedH; 
        }
        
        cv::Mat M = cv::estimateAffinePartial2D(refPoints, currPoints, cv::noArray(), ROBUST_METHOD_MOTION_ESTIMATION, MAX_RANSAC_REPROJ_THRESHOLD);

        if (!M.empty() && cv::checkRange(M)) {
            if (M.rows == 2 && M.cols == 3) {
                // Convert 2x3 matrix M = [sR | t] to 3x3 homography H
                H_prev2current = cv::Mat::eye(3, 3, CV_64F);
                M.copyTo(H_prev2current(cv::Rect(0, 0, 3, 2)));
            } else { // M is already a 3x3 homography
                M.copyTo(H_prev2current);
            }
        } 
        else{
            std::cout << "Failed to compute homography for FULL_LOCK" << std::endl;
            return previouslyReturnedH;
        }

        HomographyParameters params;
        cv::Point2d rot_center(workingSize_.width / 2.0, workingSize_.height / 2.0);

        if(decomposeHomography(H_prev2current, params, rot_center))
        {
            params.s = 1.0; // kill scaling, it is typically unstable
            H_prev2current = composeHomography(params, rot_center);

            if (REFINE_WITH_ECC) {
                cv::Mat H_prev2current_float;
                H_prev2current.convertTo(H_prev2current_float, CV_32FC1);
                H_prev2current_float.rows = 2;

                try
                {
                    cv::findTransformECC(referenceGray_, currentGray, H_prev2current_float, cv::MOTION_EUCLIDEAN,
                                    cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 0.001),
                                    cv::noArray());
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << std::endl;
                }

                H_prev2current_float.convertTo(H_prev2current_float, CV_64FC1);

                H_prev2current = cv::Mat::eye(3, 3, CV_64F);
                H_prev2current_float.copyTo(H_prev2current(cv::Rect(0, 0, 3, 2)));
            }

            previouslyReturnedH = H_prev2current.inv(); // inverts to current to previous 
        }
        
        return previouslyReturnedH;
    }
    // Add a default return for safety, though all paths should be covered.
    return cv::Mat::eye(3,3,CV_64F);
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

        // Magnify stabilized image around its center:
        cv::Mat H_stabilize_scaled_magnified = cv::Mat::eye(3, 3, CV_64F);
        H_stabilize_scaled_magnified.at<double>(0, 0) = 1.5;
        H_stabilize_scaled_magnified.at<double>(1, 1) = 1.5;
        H_stabilize_scaled_magnified.at<double>(0, 2) = 0.5 * frame.cols;
        H_stabilize_scaled_magnified.at<double>(1, 2) = 0.5 * frame.rows;

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

        // // Copy warped image onto the trail background using the mask
        // // Blur the borders of the warped_image in stabilized using an eroded version of mask so that the borders are not too sharp
        // warped_image.copyTo(stabilized, mask);
        // cv::GaussianBlur(stabilized, stabilized, cv::Size(5, 5), 0);
        // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(44, 44));
        // cv::erode(mask, mask, kernel);
        // warped_image.copyTo(stabilized, mask);


    
        auto end_warp = now();
        
        auto duration_ms_warp = toMilliseconds(start_warp, end_warp);
        
        warp_call_count_++;
        warp_avg_duration_ms_ += (duration_ms_warp - warp_avg_duration_ms_) / warp_call_count_;
    } else {
        frame.copyTo(stabilized);
    }
    
    return stabilized;
}

std::vector<cv::Point2f> Stabilizer::detectNewFeatures(const cv::Mat& gray, cv::Mat mask) {
    const int MAX_FEATURES_TO_DETECT = 1300; // Good empirical value for a working height of 360p. This helps limit computation time down the chain.
    const double QUALITY_LEVEL = 0.01;
    const int MIN_DISTANCE_720p = 10;
    const double ratio = static_cast<double>(gray.rows) / 720.0;
    const int MIN_DISTANCE = static_cast<int>(MIN_DISTANCE_720p * ratio);
    
    const int GFTT_BLOCK_SIZE = 3;
    const int GFTT_GRADIENT_SIZE = 3;
    const bool GFTT_USE_HARRIS_DETECTOR = false;
    const double GFTT_HARRIS_K = 0.04; // ignored when GFTT_USE_HARRIS_DETECTOR is false

    std::vector<cv::Point2f> featurePoints;

    cv::goodFeaturesToTrack(gray, featurePoints, 
                            MAX_FEATURES_TO_DETECT, 
                            QUALITY_LEVEL, 
                            MIN_DISTANCE, 
                            mask,
                            GFTT_BLOCK_SIZE,
                            GFTT_GRADIENT_SIZE,
                            GFTT_USE_HARRIS_DETECTOR, 
                            GFTT_HARRIS_K);
    
    auto start_gftt = now();
    cv::goodFeaturesToTrack(gray, featurePoints, MAX_FEATURES_TO_DETECT, QUALITY_LEVEL, MIN_DISTANCE);
    auto end_gftt = now();
    auto duration_ms_gftt = toMilliseconds(start_gftt, end_gftt);
    gftt_call_count_++;
    gftt_avg_duration_ms_ += (duration_ms_gftt - gftt_avg_duration_ms_) / gftt_call_count_;

    // display feature points for debugging:
    #if 0
        cv::Mat debug_image = gray.clone();
        for (const auto& point : featurePoints) {
            cv::circle(debug_image, point, 3, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Feature Points", debug_image);
        cv::waitKey(1);
    #endif
    return featurePoints;
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

cv::Mat createWarpedMask(cv::Mat presentation_image, cv::Mat H_stabilize_scaled, int border_size = 10)
{
        // Create a mask by transforming the frame corners
        cv::Mat mask = cv::Mat::zeros(presentation_image.size(), CV_8UC1);
        std::vector<cv::Point2f> original_corners(4);
        original_corners[0] = cv::Point2f(border_size, border_size);
        original_corners[1] = cv::Point2f(presentation_image.cols - border_size, border_size);
        original_corners[2] = cv::Point2f(presentation_image.cols - border_size, presentation_image.rows - border_size);
        original_corners[3] = cv::Point2f(border_size, presentation_image.rows - border_size);
        
        std::vector<cv::Point2f> transformed_corners(4);
        cv::perspectiveTransform(original_corners, transformed_corners, H_stabilize_scaled);
        
        // Convert Point2f to Point for fillConvexPoly
        std::vector<cv::Point> poly_corners(4);
        for(int i = 0; i < 4; ++i) {
            poly_corners[i] = transformed_corners[i];
        }
        
        // Draw the filled polygon on the mask
        cv::fillConvexPoly(mask, poly_corners, cv::Scalar(255));

        return mask;
}

cv::Vec2d computeTranslationShift(const cv::Vec2d& c, double s, double theta) {
    cv::Mat I = cv::Mat::eye(2, 2, CV_64F);
    cv::Mat R = (cv::Mat_<double>(2, 2) << 
                 std::cos(theta), -std::sin(theta),
                 std::sin(theta), std::cos(theta));
    cv::Vec2d t_shift = s * cv::Vec2d(cv::Mat((I - R) * cv::Vec2d(c)));
    return t_shift;
}

cv::Mat Stabilizer::copyFeathered(cv::Mat foreground, cv::Mat background_image, cv::Mat H) {

    cv::Mat warped_foreground;
    // Warp the foreground onto a canvas the size of the background

    if (foreground.size() != background_image.size()) {
        throw std::invalid_argument("Stabilizer: copyFeathered: foreground and background_image must have the same size");
    }
    if (H.size() != cv::Size(3, 3) || H.type() != CV_64F || !cv::checkRange(H)) {
        throw std::invalid_argument("Stabilizer: copyFeathered: Bad homography matrix H.");
    }

    cv::warpPerspective(foreground, warped_foreground, H, background_image.size());

    cv::Mat warped_foreground_float;
    warped_foreground.convertTo(warped_foreground_float, CV_32FC3);

    cv::Mat background_float;
    cv::Mat background_image_changed = background_image.clone();
    cv::cvtColor(background_image_changed, background_image_changed, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(background_image_changed, background_image_changed, cv::Size(7, 7), 0);
    background_image_changed *= 0.99;
    background_image_changed -= 1;
    cv::cvtColor(background_image_changed, background_image_changed, cv::COLOR_GRAY2BGR);
    background_image_changed.convertTo(background_float, CV_32FC3);

    // Create a plain white mask for the original foreground image
    //cv::Mat foreground_mask = cv::Mat(foreground.size(), CV_8UC1, cv::Scalar(255));

    // Create a mask by transforming the frame corners
    const int BORDER_SIZE = 10;
    cv::Mat foreground_mask = cv::Mat::zeros(foreground.size(), CV_8UC1);
    std::vector<cv::Point2f> original_corners(4);
    original_corners[0] = cv::Point2f(BORDER_SIZE, BORDER_SIZE);
    original_corners[1] = cv::Point2f(foreground.cols - BORDER_SIZE, BORDER_SIZE);
    original_corners[2] = cv::Point2f(foreground.cols - BORDER_SIZE, foreground.rows - BORDER_SIZE);
    original_corners[3] = cv::Point2f(BORDER_SIZE, foreground.rows - BORDER_SIZE);
    
    std::vector<cv::Point2f> transformed_corners(4);
    cv::perspectiveTransform(original_corners, transformed_corners, H);
    
    // Convert Point2f to Point for fillConvexPoly
    std::vector<cv::Point> poly_corners(4);
    for(int i = 0; i < 4; ++i) {
        poly_corners[i] = transformed_corners[i];
    }
    
    // Draw the filled polygon on the mask
    cv::fillConvexPoly(foreground_mask, poly_corners, cv::Scalar(255));

    cv::Mat warped_binary_mask;
    // Warp the mask using the same homography and to the same size
    cv::warpPerspective(foreground_mask, warped_binary_mask, H, foreground.size());


    cv::Mat feathered_alpha_mask;
    // Adjust kernel_size for more or less fuzziness. Must be an odd number.
    // Larger kernel size = more fuzzy edges.
    const int kernel_size = 101;
    cv::GaussianBlur(warped_binary_mask, feathered_alpha_mask, cv::Size(kernel_size, kernel_size), 0, 0);

    // Convert the single-channel feathered alpha mask to floating point and normalize to [0, 1]
    cv::Mat alpha_mask_float;
    feathered_alpha_mask.convertTo(alpha_mask_float, CV_32FC1, 1.0/255.0);

    // Create a 3-channel version of the alpha mask to multiply with color images
    cv::Mat alpha_mask_3channel;
    cv::cvtColor(alpha_mask_float, alpha_mask_3channel, cv::COLOR_GRAY2BGR);

    // Multiply the warped foreground with the alpha mask
    // FG_component = warped_foreground_float * alpha_mask_3channel
    cv::Mat fg_component;
    cv::multiply(alpha_mask_3channel, warped_foreground_float, fg_component);

    // Multiply the background with (1 - alpha_mask_3channel)
    // BG_component = background_float * (Scalar(1,1,1) - alpha_mask_3channel)
    cv::Mat bg_component;
    cv::Mat one_minus_alpha_3channel = cv::Scalar::all(1.0) - alpha_mask_3channel;
    cv::multiply(one_minus_alpha_3channel, background_float, bg_component);

    // Add the two components to get the final blended image
    cv::Mat blended_image_float;
    cv::add(fg_component, bg_component, blended_image_float);

    // Convert the result back to CV_8UC3 if needed
    cv::Mat final_result;
    blended_image_float.convertTo(final_result, CV_8UC3);

    return final_result;
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
        prevPoints_ = detectNewFeatures(gray);
        prevGray_ = gray.clone();
        return frame;
    }

    std::vector<cv::Point2f> filtered_previousPoints;
    std::vector<cv::Point2f> filtered_currentPoints;

    trackFeatures(prevGray_, gray, prevPoints_, filtered_previousPoints, filtered_currentPoints);

    int filtered_currentpoints_percentage = static_cast<int>(100.0 * static_cast<double>(filtered_currentPoints.size()) / static_cast<double>(prevPoints_.size()));
    //std::cout << "filtered_previousPoints.size() = " << filtered_previousPoints.size() << " (" << filtered_currentpoints_percentage     << "%)" << std::endl;
    //std::cout << "filtered_currentPoints.size() = " << filtered_currentPoints.size() << " (" << filtered_currentpoints_percentage << "%)" << std::endl;

    cv::Mat H_prev2curr = estimateMotion(filtered_previousPoints, filtered_currentPoints);
    
    // Get current frame index
    uint64_t current_idx = stabilizationWindow_.frames.back().frame_idx;
    
    // Update transformations window
    updateTransformations(H_prev2curr, current_idx);

    // These two would be be false if both totalPastFrames_ and totalFutureFrames_ were 0.
    assert(stabilizationWindow_.frames.size() >= 2);
    assert(stabilizationWindow_.transformations.size() >= 1); // at least one transformation between the first and last frame

    assert(stabilizationWindow_.frames.size() == stabilizationWindow_.transformations.size() + 1); // one more frame than transformations
    assert(stabilizationWindow_.frames.front().frame_idx == stabilizationWindow_.transformations.front().from_frame_idx);
    assert(stabilizationWindow_.frames.back().frame_idx == stabilizationWindow_.transformations.back().to_frame_idx);

    // Determine which frame to present
    size_t presentation_frame_idx = 0;
    if (stabilizationWindow_.frames.size() > totalFutureFrames_) {
        presentation_frame_idx = stabilizationWindow_.frames.size() - totalFutureFrames_ - 1;
    }

    cv::Mat H_stabilize;

    const double epsilon = 1e-6;
    
    cv::Mat H_global_smoothing = calculateGlobalSmoothingStabilization(presentation_frame_idx);
    cv::Mat H_lock = calculateFullLockStabilization(presentation_frame_idx);
    
    cv::Point2d rot_center = cv::Point2d(workingSize_.width / 2.0, workingSize_.height / 2.0);
    
    HomographyParameters homography_params_lock;
    if(!decomposeHomography(H_lock, homography_params_lock))
    {
        std::cerr << "Error: Failed to decompose homography" << std::endl;
        H_lock = cv::Mat::eye(3, 3, CV_64F);
    }
    
    cv::Mat R = cv::getRotationMatrix2D(rot_center, homography_params_lock.theta * 180.0 / M_PI, 1.0);

    // Augment R with a (0,0,1) row at the bottom
    cv::Mat R_augmented = cv::Mat::eye(3, 3, CV_64F);
    R.copyTo(R_augmented(cv::Rect(0, 0, 3, 2)));
    R_augmented.at<double>(2, 2) = 1.0;

    cv::Mat H_translation_lock = R_augmented * H_lock; // basically we're re-adding the accumulated rotation to the lock transform, so only the translation is left locked

    cv::Mat H_rotation_lock = R_augmented.inv(); // we're only cancelling the accumulated rotation, so the translation is left free

    HomographyParameters homography_params_stabilize;
    cv::Vec2d t_shifted;

    // Calculate stabilization transform based on selected mode
    switch (stabilizationMode_) {
        case StabilizationMode::ACCUMULATED_FULL_LOCK:
            H_stabilize = H_lock; // simply use the full lock transform
            break;
        case StabilizationMode::TRANSLATION_LOCK:
            H_stabilize = H_translation_lock;
            break;
        case StabilizationMode::ROTATION_LOCK:
            H_stabilize = H_rotation_lock;
            break;
        case StabilizationMode::GLOBAL_SMOOTHING:
            H_stabilize = H_global_smoothing;
            break;
        case StabilizationMode::ORB_FULL_LOCK:
            H_stabilize = H_lock;
            break;
        case StabilizationMode::SIFT_FULL_LOCK:
            H_stabilize = H_lock;
            break;
        default:
            throw std::invalid_argument("Stabilizer: Invalid stabilization mode");
            break;
    }
    
    // Scale stabilization transform to match original resolution
    cv::Mat H_stabilize_scaled = H_stabilize.clone();
    if (std::abs(scaleFactor_ - 1.0) > epsilon) {
        // Adjust translation components
        H_stabilize_scaled.at<double>(0, 2) /= scaleFactor_;
        H_stabilize_scaled.at<double>(1, 2) /= scaleFactor_;
    }
    

    cv::Mat presentation_image = stabilizationWindow_.frames[presentation_frame_idx].image;

    cv::Mat presentation_output;
    
    #if 0
        presentation_output = copyFeathered(presentation_image, trail_background_, H_stabilize_scaled);
        trail_background_ = presentation_output.clone();
    #else
        cv::Scalar avg_color = 0.5 * cv::mean(presentation_image);
        cv::Mat warped_image;
        cv::warpPerspective(presentation_image, warped_image, H_stabilize_scaled, frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, avg_color);
        presentation_output = warped_image;
    #endif
    
    // Detect new features for next frame
    prevPoints_ = detectNewFeatures(gray);
    prevGray_ = gray.clone();
    
    // Print timing information periodically
    //printTimings(); // Keep this commented out when not debugging. Don't remove.
    
    return presentation_output;
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
    // Don't confuse Q with R. Q is the orthogonal matrix, R is the upper triangular matrix.
    // Elsewhere, R is typically used to denote a rotation matrix, not here.
    // We wanted to follow the Math convention of using Q, R labels for QR decomposition.

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

    // We don't check that Q is a rotation matrix or a reflection matrix, as we don't care about that here
    // as it makes sense to accomodate both cases inside this function.
    // We should check that Q is a rotation matrix elsewhere, where futher context matters, not here.
    // Here, inside this method, we are only interested in the general QR decomposition.

    // Assert that R is upper triangular
    R.at<double>(1, 0) = 0.0; // R[1, 0] should be 0
}


bool Stabilizer::decomposeHomography(const cv::Mat& H, HomographyParameters &params_out, cv::Point2d rot_center) {
    if (H.empty() || H.rows != 3 || H.cols != 3 || H.type() != CV_64F) {
        throw std::invalid_argument("Error: Input homography matrix must be a non-empty 3x3 CV_64F matrix.");
    }

    const double epsilon = 1e-6;
    cv::Mat H_norm = cv::Mat::eye(3, 3, CV_64F);

    if (!cv::checkRange(H))
    {
        std::cerr << "Error: H matrix contains non-finite or NaN entries." << std::endl;
        return false;
    }

    // Normalize H such that H(2, 2) = 1
    double h33 = H.at<double>(2, 2);
    if (std::abs(h33) < epsilon) {
        std::cerr << "Error: H(2, 2) is close to zero. Degenerate homography." << std::endl;
        return false;
    }
    else {
        H_norm = H / h33;
    }

    // Extract translation vector t = [tx, ty] and projective vector v = [v1, v2]
    cv::Mat t = H_norm(cv::Rect(2, 0, 1, 2));
    cv::Mat v = H_norm(cv::Rect(0, 2, 2, 1));

    // Extract A matrix from H_norm
    cv::Mat A = H_norm(cv::Rect(0, 0, 2, 2));

    // Extract RK matrix and scaling factor s from A
    cv::Mat sRK = A - t * v;

    if (!cv::checkRange(sRK)) {
        std::cerr << "Error: sRK matrix contains non-finite or NaN entries." << std::endl;
        return false;
    }

    double det_sRK = cv::determinant(sRK);
    if (std::isnan(det_sRK) || std::isinf(det_sRK) || det_sRK < 0 || std::abs(det_sRK) < epsilon) {
        std::cerr << "Error: determinant of sRK is close to zero. Degenerate homography." << std::endl;
        return false;
    }
    double s = std::sqrt(det_sRK); // sqrt because A is 2x2 matrix
    cv::Mat RK = sRK / s;

    // Extract R and K from RK using QR-decomposition
    // They are both 2x2 matrices, and there is a closed-form solution for them
    // R = [cos(theta) -sin(theta); sin(theta) cos(theta)], det(R) = 1, R is a rotation matrix, R^T = R^-1
    // K = [k1 d; 0 k2], k1*k2 = det(K) = 1, K is a upper triangular matrix
    cv::Mat R, K;
    qrDecomposition2x2(RK, R, K);

    if (!cv::checkRange(R) || !cv::checkRange(K)) {
        std::cerr << "Error: R or K matrix contains non-finite or NaN entries." << std::endl;
        return false;
    }

    // Check that R is a rotation matrix, not a reflection matrix
    double det_R = cv::determinant(R);
    if (std::abs(det_R - 1.0) > epsilon) {
        std::cerr << "Error: R is not a rotation matrix. Det(R) = " << det_R << ", should be 1." << std::endl;
        return false;
    }

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
    cv::Vec2d t_vec = cv::Vec2d(t);
    cv::Vec2d t_shifted = t_vec - t_shift;

    // only need k1, as k2 = 1/k1
    params_out = HomographyParameters{s, theta, k1, delta, t_shifted, v};
    return true;
}

cv::Mat Stabilizer::composeHomography(const HomographyParameters& params, cv::Point2d rot_center) {
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
    cv::Vec2d t_shift = params.s * cv::Vec2d(cv::Mat((I - R) * cv::Vec2d(rot_center)));
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
