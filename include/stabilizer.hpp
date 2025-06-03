#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <chrono>
#include <cstddef> // Keep for size_t
#include <cstdint> // Keep for uint64_t

/**
 * @brief Enumeration of available video stabilization modes.
 * 
 * Defines different approaches to video stabilization:
 * - ACCUMULATED_FULL_LOCK: Eliminates all camera motion by tracking feature points 
 *                          across consecutive frames, computing frame-to-frame 
 *                          transformation matrices, and chaining them relative to a
 *                          reference frame.
 * - ORB_FULL_LOCK: Eliminates all camera motion by directly aligning each frame
 *                  to a reference frame using ORB feature matching and
 *                  transformation matrix fitting.
 * - SIFT_FULL_LOCK: Same as ORB_FULL_LOCK but uses SIFT features for more robust
 *                   matching at the cost of higher computation time.
 * @todo fix partial locking modes as they are not correctly implemented yet.
 * - TRANSLATION_LOCK: Compensates only for horizontal and vertical camera movement.
 * - ROTATION_LOCK: Compensates only for camera rotation around the image centre.
 * - GLOBAL_SMOOTHING: Reduces camera shake by averaging the motion between the
 *                     presented frame and its neighbouring frames within a sliding
 *                     temporal window, preserving intentional camera motion while
 *                     removing high-frequency jitter.
 */
enum class StabilizationMode {
    ACCUMULATED_FULL_LOCK,
    ORB_FULL_LOCK,
    SIFT_FULL_LOCK,
    TRANSLATION_LOCK,
    ROTATION_LOCK,
    GLOBAL_SMOOTHING
};

/**
 * @brief Parameters of a 2D homography. These components are uniquely mapped to and
 * from a 2D homography matrix.
 */
struct HomographyParameters {
    double s{1.0};       ///< Isotropic scaling factor
    double theta{0.0};   ///< In-image-plane rotation angle
    double k{1.0};       ///< Anisotropy scaling ratio (k1=k, k2=1/k1 so that k1*k2=1)
    double delta{0.0};   ///< Shear value
    cv::Vec2d t{0.0,0.0};    ///< Translation vector
    cv::Vec2d v{0.0,0.0};    ///< Horizon line shift vector

    /**
     * @brief Returns a string representation of the parameters for debugging.
     * @return String containing all parameter values
     */
    std::string print() const {
        return "s: " + std::to_string(s) + ", theta: " + std::to_string(theta) + ", k: " + std::to_string(k) + ", delta: " + std::to_string(delta) + ", t: " + std::to_string(t[0]) + ", " + std::to_string(t[1]) + ", v: " + std::to_string(v[0]) + ", " + std::to_string(v[1]);
    }
};

/**
 * @brief Represents a transformation between two frames.
 * 
 * Stores a 3x3 homography matrix and the frame indices it transforms between.
 */
struct Transformation {
    cv::Mat H;                 ///< 3x3 homography matrix
    size_t from_frame_idx{0};  ///< Source frame index
    size_t to_frame_idx{0};    ///< Target frame index
    
    /**
     * @brief Computes the inverse transformation.
     * @return New Transformation object representing the inverse
     */
    Transformation inverse() const {
        cv::Mat inv;
        cv::invert(H, inv);
        return Transformation{inv, to_frame_idx, from_frame_idx};
    }
};

/**
 * @brief Represents a video frame with its index.
 */
struct Frame {
    cv::Mat image;       ///< Frame image data, in OpenCV BGR colour format
    size_t frame_idx{};  ///< Frame index in the sequence of frames
};

/**
 * @brief Maintains a sliding window of neighbouring frames and the transformations
 * between them. Used for stabilization calculations.
 */
struct StabilizationWindow {
    std::deque<Transformation> transformations;  ///< Queue of frame-to-frame transformations
    std::deque<Frame> frames;                    ///< Queue of frames
};

/**
 * @brief Main class for video stabilization.
 * 
 * Implements various video stabilization algorithms using different motion
 * estimation and compensation techniques. Supports multiple stabilization
 * modes and real-time processing.
 */
class Stabilizer {
public:
    /// Type alias for millisecond duration
    using milli_duration = std::chrono::duration<double, std::milli>;
    
    /**
     * @brief Gets the current time point.
     * @return Current high resolution clock time point
     */
    inline auto now() {
        return std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Constructs a Stabilizer with specified temporal window parameters.
     * 
     * Initializes a video stabilizer that maintains a sliding temporal window for
     * motion analysis and stabilization. The stabilizer buffers futureFrames ahead
     * of the presented frame to incorporate upcoming motion information, which helps
     * minimize visual lag in the stabilized output, at the cost of delayed presentation.
     * A longer temporal window (more pastFrames and futureFrames) results in smoother
     * stabilization since more frames are used to analyze and compensate for motion.
     * 
     * @param pastFrames Number of historical frames to keep for motion analysis
     * @param futureFrames Number of upcoming frames to buffer before producing a
     *                     stabilized output frame
     * @param workingHeight Internal processing height in pixels (preserves aspect
     *                      ratio)
     * @throws std::invalid_argument if both pastFrames and futureFrames are 0, or if
     *         workingHeight is not between 90 and 2160 rows
     */
    Stabilizer(size_t pastFrames = 15, size_t futureFrames = 15, int workingHeight = 360);
    
    /**
     * @brief Processes an input frame and returns a stabilized frame.
     * 
     * This method implements different stabilization modes:
     * 
     * For GLOBAL_SMOOTHING mode:
     * - The input frame is added to a sliding window of frames.
     * - Motion is analyzed using the input frame for feature tracking.
     * - A presentation frame is selected from the window (delayed by totalFutureFrames_)
     * - The presentation frame is stabilized using averaged motion information from
     *   the temporal window.
     * 
     * For FULL_LOCK modes:
     * - The input frame is added to a sliding window of frames
     * - Motion is analyzed relative to a fixed past reference frame
     * - A presentation frame is selected from the window (delayed by totalFutureFrames_)
     * - The presentation frame is stabilized to maintain alignment with the reference
     * 
     * @note For the first frame in any mode, returns the input frame directly since
     *       no history exists.
     * 
     * @note Due to the temporal window buffering, the returned frame is not the
     *       stabilized version of the input frame, but rather a stabilized frame
     *       from totalFutureFrames_ frames ago in the temporal window.
     * 
     * @param frame Input frame to add to the processing window
     * @return Stabilized presentation frame based on the current stabilization mode
     * @throws std::invalid_argument if frame has invalid size
     */
    cv::Mat stabilizeFrame(const cv::Mat& frame);

    /**
     * @brief Sets the stabilization mode.
     * 
     * When switching stabilization modes, the following occurs:
     * 1. The new mode is applied immediately to the next frame
     * 2. The existing frame window and motion history are preserved
     * 3. For locking modes, the next presentation frame processed becomes the new
     *    reference frame that subsequent frames lock to
     * 4. For GLOBAL_SMOOTHING mode, motion smoothing calculations immediately begin
     *    using the sliding temporal window.
     * 
     * Mode transitions are seamless with no temporal discontinuities in the output,
     * as frame timing and ordering remain consistent across all stabilization modes.
     * While the frame position within the presentation canvas may shift during mode
     * transitions, the temporal sequence remains uninterrupted.
     * 
     * @param mode New stabilization mode to use, one of @ref StabilizationMode values.
     */
    void setStabilizationMode(StabilizationMode mode);

    /**
     * @brief Returns the total size of the frame window.
     * @return Total number of frames in the stabilization window
     * @note This is the total number of frames in the stabilization window, including
     *       the current frame, the past frames, and the future frames.
     */
    inline size_t totalFrameWindowSize() const {
        return totalPastFrames_ + 1 + totalFutureFrames_;
    }

    /**
     * @brief Converts a time duration to milliseconds.
     * @param start Start time point
     * @param end End time point
     * @return Duration in milliseconds
     */
    inline milli_duration toMilliseconds(
                            const std::chrono::high_resolution_clock::time_point& start,
                            const std::chrono::high_resolution_clock::time_point& end) {
        return std::chrono::duration_cast<milli_duration>(end - start);
    }
    
    /**
     * @brief Decomposes a homography matrix into its constituent parameters.
     * This decomposition is mathematically unique.
     * 
     * Decomposes a 3x3 homography matrix into its constituent motion parameters, as
     * defined in the @ref HomographyParameters struct.
     * 
     * @param H Input 3x3 homography matrix (must be CV_64F)
     * @param params_out Output parameters structure
     * @param rot_center Point around which rotation and scaling transformations are
     *                   applied
     * @return true if decomposition succeeded, false if matrix is degenerate
     * @throws std::invalid_argument if H is not a valid 3x3 CV_64F matrix
     * @note params_out is left unchanged if the decomposition fails.
     */
    static bool decomposeHomography(const cv::Mat& H, HomographyParameters &params_out, 
                                    cv::Point2d rot_center = cv::Point2d(0,0));

    /**
     * @brief Composes a homography matrix from constituent parameters.
     * This composition is mathematically unique.
     * 
     * Creates a 3x3 homography matrix from the constituent motion parameters, as
     * defined in the @ref HomographyParameters struct.
     * 
     * @param params Input parameters structure
     * @param rot_center Point around which rotation and scaling transformations are
     *                   applied
     * @return 3x3 homography matrix (CV_64F)
     */
    static cv::Mat composeHomography(const HomographyParameters& params, 
                                     cv::Point2d rot_center = cv::Point2d(0,0));

private: // helper methods for the main stabilizeFrame() method

    /**
     * @brief Blends a warped foreground image with a background using feathered edges.
     * 
     * Warps the foreground image using the provided homography, creates a feathered
     * mask from the transformed corners, and blends with the background for smooth
     * edge transitions.
     * 
     * @note This method is currently unused in the implementation. The actual
     *       warping is performed directly in stabilizeFrame() using cv::warpPerspective.
     * 
     * @note This computationally intensive method is currently unused in production.
     *       It is retained for potential future use in offline video processing
     *       or GPU-accelerated implementations.
     * 
     * @param foreground Foreground image to warp and blend
     * @param background_image Background image for blending
     * @param H Homography matrix for warping the foreground
     * @return Blended image with feathered edges
     * @throws std::invalid_argument if images have different sizes or H is invalid
     */
    static cv::Mat copyFeathered(const cv::Mat foreground, const cv::Mat background_image, 
                                 const cv::Mat H);

    /**
     * @brief Calculates stabilization transform for full lock modes.
     * 
     * Implements different strategies based on stabilization mode:
     * - ACCUMULATED_FULL_LOCK: Accumulates frame-to-frame transformations and returns inverse
     * - ORB_FULL_LOCK: Uses ORB features to compute motion relative to a reference frame
     * - SIFT_FULL_LOCK: Uses SIFT features to compute motion relative to a reference frame
     * - Returns identity matrix for GLOBAL_SMOOTHING mode
     * 
     * For ORB/SIFT modes, establishes a reference frame on first call and computes
     * transformations to cancel motion relative to that reference.
     * 
     * @param presentation_frame_idx Index of presentation frame in the stabilization window
     * @return 3x3 homography matrix for full motion compensation
     */
    cv::Mat calculateFullLockStabilization(size_t presentation_frame_idx);
    
    /**
     * @brief Calculates stabilization transform for global smoothing mode.
     * 
     * Computes a smoothing transformation by averaging motion over the temporal window
     * to reduce camera shake while preserving intentional camera movement.
     * 
     * The algorithm works by:
     * 1. Computing frame-to-frame transformations across the window, fitting a
     *    rigid-body motion model to sparse optical flow between consecutive frames.
     * 2. Computing an average of the transformations between the current frame and each
     *    of the frames in the window.
     * 3. The returned homography matrix is the average of the transformations between
     *    the current frame and each of its neighbouring frames in the window. Transforming
     *    the current frame with this homography matrix results in a stabilized frame.
     * 
     * This approach allows for preservation of intentional camera motion through
     * temporal averaging.
     * 
     * @note Although homography matrices can represent general perspective transforms,
     *       we constrain our frame-to-frame motion estimation to simpler rigid-body
     *       models. This choice improves stabilization accuracy since rigid transforms
     *       are more robust against image noise such as motion blur that commonly occur
     *       with longer camera shutter exposures. The reduced complexity of rigid-body
     *       models leads to more reliable motion estimation between consecutive frames.
     *       The homography matrices provide a more general mathematical representation
     *       for describing the motion transformations between consecutive frames, from
     *       which rigid-body motion is purely a special case.
     * 
     * @param presentation_frame_idx Index of presentation frame in the stabilization window
     * @return 3x3 homography matrix for motion smoothing
     */
    cv::Mat calculateGlobalSmoothingStabilization(size_t presentation_frame_idx);

    /**
     * @brief Detects corner features in a grayscale image using OpenCV's goodFeaturesToTrack.
     * 
     * Detects strong corner features suitable for tracking, with parameters
     * scaled based on image resolution. Includes performance timing measurement.
     * 
     * NOTE: Current implementation has a bug - the mask parameter is ignored
     * due to duplicate goodFeaturesToTrack calls where timing is measured on 
     * the second call that doesn't use the mask. 
     * @todo fix this.
     * 
     * @param gray Input grayscale image
     * @param mask Optional mask for feature detection (currently ignored due to implementation bug)
     * @return Vector of detected corner feature points
     */
    std::vector<cv::Point2f> detectNewFeatures(const cv::Mat& gray, cv::Mat mask = cv::Mat());


    /**
     * @brief Initializes frame processing and sets up working resolution.
     * @param frame Input frame to initialize
     * @throws std::invalid_argument if frame has invalid size
     */
    void initializeFrame(const cv::Mat& frame);

    /**
     * @brief Prepares the background for trail effect rendering.
     * 
     * Currently returns a clone of the existing trail background. Most processing
     * operations (resizing, blurring, darkening) are commented out in the implementation.
     * @todo bring back the trail effect rendering.
     * @return Cloned trail background image
     */
    cv::Mat prepareTrailBackground();

    /**
     * @brief Adds a frame to the stabilization window and manages window size.
     * @param frame Frame to add to the sliding window
     */
    void addFrameToWindow(const cv::Mat& frame);
    
    /**
     * @brief Tracks features between consecutive frames using sparse optical flow.
     * 
     * Uses Lucas-Kanade optical flow to track feature points from the previous
     * frame to the current frame, filtering out unsuccessfully tracked points.
     * 
     * @param previous_gray Previous frame in grayscale
     * @param current_gray Current frame in grayscale
     * @param previousPoints Features from previous frame to track
     * @param filtered_previousPoints Output: successfully tracked features from previous frame
     * @param filtered_currentPoints Output: corresponding tracked positions in current frame
     */
    void trackFeatures(const cv::Mat& previous_gray, const cv::Mat& current_gray, 
                        const std::vector<cv::Point2f> & previousPoints,
                        std::vector<cv::Point2f> & filtered_previousPoints,
                        std::vector<cv::Point2f> & filtered_currentPoints);

    /**
     * @brief Estimates motion between two sets of corresponding points.
     * 
     * Computes motion using cv::estimateAffinePartial2D (partial 2D Euclidean transform:
     * isotropic scaling, rotation, translation) by default, with RANSAC for robustness.
     * Full homography estimation is available but disabled (#if 0). Removes isotropic 
     * scaling from the result, yielding rigid-body motion model for enhanced stability
     * and returns identity matrix if insufficient points or estimation fails.
     * 
     * @param previousPoints Corresponding points from previous frame
     * @param currentPoints Corresponding points from current frame
     * @return 3x3 homography matrix representing the motion (identity if estimation fails)
     */
    cv::Mat estimateMotion(const std::vector<cv::Point2f>& previousPoints, 
                           const std::vector<cv::Point2f>& currentPoints);
    
    /**
     * @brief Updates the transformation history with new motion and manages window size.
     * 
     * Adds the current frame-to-frame transformation to the sliding window and
     * removes old transformations to maintain the window size constraints.
     * 
     * @param H_prev2curr Homography from previous frame to current frame
     * @param current_idx Current frame index
     */
    void updateTransformations(const cv::Mat& H_prev2curr, uint64_t current_idx);
    
    /**
     * @brief Warps the presentation frame using the stabilization transform.
     * 
     * NOTE: This method is currently unused in the implementation. The actual
     * warping is performed directly in stabilizeFrame() using cv::warpPerspective.
     * 
     * @todo bring back the trail effect rendering, and unify the simpler rendering and
     *       the trail effect into this single method.
     * 
     * @param frame Input frame (used for size reference)
     * @param H_stabilize_scaled Stabilization homography matrix
     * @param next_trail_background Background for trail effect
     * @param presentation_frame_idx Index of frame to warp from the stabilization window
     * @return Warped and stabilized presentation frame
     */
    cv::Mat warpFrame(const cv::Mat& frame, const cv::Mat& H_stabilize_scaled, 
                      const cv::Mat& next_trail_background, size_t presentation_frame_idx);
    
    /**
     * @brief Prints timing statistics for computation latency monitoring.
     * 
     * @todo bring back the timing statistics printing, setting a runtime option to enable/disable it.
     */
    void printTimings();

    size_t totalPastFrames_{0};      ///< Number of past frames in stabilization window
    size_t totalFutureFrames_{0};    ///< Number of future frames in stabilization window
    int workingHeight_{360};         ///< Target height for internal processing
    double scaleFactor_{1.0};        ///< Scale factor between original and working resolution
    cv::Size original_frame_size_{0, 0};  ///< Original frame size
    cv::Size workingSize_{0, 0};     ///< Working frame size
    
    StabilizationWindow stabilizationWindow_;  ///< Sliding window of frames and transformations
    
    cv::Mat prevGray_;               ///< Previous frame in grayscale, in working resolution/size
    std::vector<cv::Point2f> prevPoints_;  ///< Features from previous frame

    cv::Mat trail_background_;       ///< Background for trail effect

    Transformation accumulatedTransform_;  ///< Accumulated transformation for full lock mode

    // ORB-based motion estimation variables
    int64_t referenceFrameIdx_{-1};  ///< Reference frame index for full lock mode
    cv::Mat referenceGray_;          ///< Reference frame grayscale image
    std::vector<cv::KeyPoint> orb_referenceKeypoints_;  ///< Reference frame ORB keypoints
    cv::Mat orb_referenceDescriptors_;  ///< Reference frame ORB descriptors

    std::vector<cv::KeyPoint> sift_referenceKeypoints_;  ///< Reference frame SIFT keypoints
    cv::Mat sift_referenceDescriptors_;  ///< Reference frame SIFT descriptors

    cv::Ptr<cv::ORB> orb_;           ///< ORB feature detector
    cv::Ptr<cv::SIFT> sift_;         ///< SIFT feature detector

    std::vector<cv::Point2f> gftt_reference_corners_;  ///< Reference frame corners for GFTT (goodFeaturesToTrack)

    StabilizationMode stabilizationMode_{StabilizationMode::GLOBAL_SMOOTHING};  ///< Current stabilization mode
    // Timing variables (rolling averages)
    milli_duration gftt_avg_duration_ms_{0.0};         ///< Average duration for GFTT
    long long gftt_call_count_{0};                     ///< Number of GFTT calls (goodFeaturesToTrack)
    milli_duration lk_avg_duration_ms_{0.0};           ///< Average duration for Lucas-Kanade optical flow
    long long lk_call_count_{0};                       ///< Number of Lucas-Kanade optical flow calls
    milli_duration homography_avg_duration_ms_{0.0};   ///< Average duration for homography estimation
    long long homography_call_count_{0};               ///< Number of homography estimations
    milli_duration warp_avg_duration_ms_{0.0};         ///< Average duration for warping
    long long warp_call_count_{0};                     ///< Number of warping operations
    

    // Timing print variables
    std::chrono::high_resolution_clock::time_point last_print_time_;    ///< Last time statistics were printed
    const std::chrono::milliseconds print_interval_{1000};         ///< Interval between statistics prints
};
