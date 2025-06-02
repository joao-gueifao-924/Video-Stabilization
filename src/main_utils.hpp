#pragma once

#include "stabilizer.hpp"
#include "camera_engine.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>
#include <memory>
#include <string>

namespace VideoStabilizer {

enum class InputMode {
  UNSPECIFIED,
  SIMULATOR,
  CAMERA,
  FILE
};

struct InputConfig {
  // Defaults overridden by optional command line arguments
  InputMode mode = InputMode::UNSPECIFIED;
  std::string path;  // Unified path field for video files or simulator images
  int cameraId = 0;                // Camera ID for --camera mode
  double pastWindowSecs = 2.0;     // Past stabilization window in seconds
  double futureWindowSecs = 1.5;   // Future stabilization window in seconds  
  int workingHeight = 360;         // Stabilizer working height in pixels
};

/**
 * Parses command line arguments and populates the InputConfig structure.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param config Output parameter that will store the parsed configuration
 * 
 * @return true if arguments were parsed successfully, false otherwise
 * 
 * @note Validates:
 *       - Exactly one input mode (--simulator, --camera, or --file) is specified
 *       - Required values are provided for the chosen mode
 *       - Optional parameters (--past-window, --future-window, --working-height) 
 *         have valid values
 *       - Total window size (past + future) meets minimum requirements
 * 
 * @throws std::invalid_argument If numeric parameters contain invalid values
 * @throws std::out_of_range If numeric parameters are outside valid ranges
 */
bool parseCommandLineArgs(int argc, char* argv[], InputConfig& config);

/**
 * Prints usage information and command line options to stdout.
 * 
 * @param programName Name of the program executable
 * 
 * @note Displays:
 *       - Program usage syntax
 *       - Available input modes
 *       - Optional stabilizer parameters
 *       - Example commands
 */
void printUsage(const char* programName);

/**
 * Initializes the video input source based on the provided configuration.
 * 
 * @param config Configuration specifying input mode and related parameters
 * @param fps Output parameter that will store the frames per second of the video source
 * @param cameraEngine Shared pointer to CameraEngine, initialised only in simulator mode
 * @param cap OpenCV VideoCapture object for camera/file input modes (uninitialized if in simulator mode)
 * @param default_camera_params Default camera parameters used to initialise simulator
 * 
 * @return true if initialization succeeds, false if any errors occur:
 *         - Camera mode: Failed to open camera or invalid camera ID
 *         - File mode: Failed to open video file
 *         - Simulator mode: Failed to initialize CameraEngine
 * 
 * @note For camera mode:
 *       - Sets resolution to 1280x720
 *       - Defaults to 30 fps if camera returns 0 fps
 * @note For file mode:
 *       - Defaults to 30 fps if video file returns invalid fps
 * @note For simulator mode:
 *       - Uses fixed 30 fps
 *       - Creates a floor texture with a tile width of 10.0 metres in the simulated 3D space
 *       - Initializes camera with provided default parameters
 *       - Refer to CameraEngine for more details on the floor texture composition
 * 
 * @throws std::runtime_error If CameraEngine initialization fails in simulator mode
 *                           or if floor texture image cannot be loaded
 * @throws cv::Exception If OpenCV encounters an internal error during VideoCapture operations
 */
bool initializeInputSource(const InputConfig& config, double& fps, 
                          std::shared_ptr<CameraEngine> cameraEngine, 
                          cv::VideoCapture& cap, 
                          const CameraEngine::CameraParams& default_camera_params);

/**
 * Sets up the video stabilizer with specified parameters and creates display windows.
 * 
 * @param past_frames Number of frames before the current frame to analyze for stabilization.
 *                   Higher values provide more historical context for smoother motion.
 * @param future_frames Number of frames after the current frame to analyze for stabilization.
 *                     Higher values improve smoothing and reduce lag by providing more future context
 *                     for motion prediction. However, this increases video presentation delay.
 *                     Total frames (past + future) determines the stabilization window size.
 * @param working_height Height in pixels for internal stabilizer processing. Higher values may improve 
 *                      stabilization accuracy but increase computation time.
 * 
 * @return Configured Stabilizer object ready for frame processing
 * 
 * @note Creates the following OpenCV windows:
 *       - "Original" for displaying unstabilized input frames
 *       - "Stabilized" for displaying processed stabilized frames
 * 
 * @throws cv::Exception If OpenCV window creation fails
 * @throws std::invalid_argument If working_height is zero or negative
 */
Stabilizer setupStabilizerAndWindows(int past_frames, int future_frames, 
                                    int working_height);

/**
 * Processes keyboard input for camera movement control in simulator mode.
 * 
 * @param key ASCII code of the pressed key
 * @param cameraEngine Shared pointer to the CameraEngine for simulator control
 * @param default_camera_params Default camera parameters for reset functionality
 * 
 * @return true if a camera movement command was processed, false otherwise
 * 
 * @note Only used when InputMode is SIMULATOR
 * 
 * @note Supported keyboard controls:
 *       - 'W'/'S': Move camera forward/backward
 *       - 'A'/'D': Move camera left/right  
 *       - 'Q'/'E': Roll camera counter-clockwise/clockwise
 *       - 'C': Move camera down
 *       - 'SPACE': Move camera up
 *       - 'P': Reset camera to default position and orientation
 * 
 * @throws std::runtime_error If camera movement update fails in CameraEngine
 */
bool handleCameraMovement(int key, std::shared_ptr<CameraEngine> cameraEngine, 
                         const CameraEngine::CameraParams& default_camera_params);

/**
 * Processes keyboard input for stabilization parameter controls.
 * 
 * @param key ASCII code of the pressed key
 * @param stabilizer Reference to the Stabilizer object to be controlled
 * 
 * @note Supported keyboard controls:
 *       - 'X': Reset stabilizer (to Global Smoothing)
 *       - 'F': Full lock stabilization (Optical Flow based)
 *       - 'O': ORB-based Full lock stabilization
 *       - 'L': SIFT-based Full lock stabilization
 *       - 'T': Translation lock stabilization mode
 *       - 'R': Rotation lock stabilization mode
 * 
 * @throws std::out_of_range If stabilization parameters exceed valid ranges
 */
void handleStabilizationControls(int key, Stabilizer& stabilizer);

/**
 * Captures a single frame from the configured input source.
 * 
 * @param config Configuration specifying the input mode and parameters
 * @param cap OpenCV VideoCapture object for camera/file input (unused in simulator mode)
 * @param cameraEngine Shared pointer to CameraEngine for simulator input (unused in other modes)
 * @param frame Output parameter that will store the captured frame
 * 
 * @return true if frame capture succeeds, false if capture fails or end of input reached
 * 
 * @note Behavior varies by input mode:
 *       - CAMERA/FILE: Uses OpenCV VideoCapture to read next frame
 *       - SIMULATOR: Renders frame using CameraEngine
 *       - Frame is stored in BGR format for OpenCV compatibility
 * 
 * @throws cv::Exception If OpenCV frame capture encounters an internal error
 * @throws std::runtime_error If CameraEngine frame rendering fails in simulator mode
 */
bool captureFrame(const InputConfig& config, cv::VideoCapture& cap,
                 std::shared_ptr<CameraEngine> cameraEngine, cv::Mat& frame);

/**
 * Adds informational overlays and metadata to the captured frame.
 * 
 * @param frame Input/output frame to which overlays will be added
 * @param config Configuration containing input mode and stabilizer parameters
 * @param cameraEngine Shared pointer to CameraEngine for simulator-specific overlays
 * @param fps Current frames per second for performance overlay
 * 
 * @throws cv::Exception If OpenCV text rendering operations fail
 */
void addFrameOverlays(cv::Mat& frame, const InputConfig& config, 
                     std::shared_ptr<CameraEngine> cameraEngine, double fps);

/**
 * Processes frames through the stabilizer and displays results in OpenCV windows.
 * 
 * @param frame Input frame to be processed and displayed
 * @param stabilizer Reference to the Stabilizer object that performs stabilization
 * @param originalFrameBuffer Buffer storing original, non-stabilised frames for synchronized display
 * @param future_frames Number of future frames in stabilizer window
 * @param config Input configuration parameters
 * @param cameraEngine Simulator camera engine (nullptr if not in simulator mode)
 * @param start_time Iteration start time for FPS calculation
 * 
 * @note Manages frame buffers and displays:
 *       1. Adds frame to stabilizer queue
 *       2. Stores original frame for later display
 *       3. Gets stabilized frame if available
 *       4. Shows original and stabilized frames side-by-side
 *       5. Updates FPS counter and overlays
 * 
 * @throws cv::Exception If frame display fails
 * @throws std::runtime_error If stabilizer processing fails
 */
void processAndDisplayFrames(cv::Mat frame, Stabilizer& stabilizer, 
                           std::deque<cv::Mat>& originalFrameBuffer, 
                           int future_frames, const InputConfig& config, 
                           std::shared_ptr<CameraEngine> cameraEngine, 
                           std::chrono::high_resolution_clock::time_point start_time);

/**
 * Performs cleanup operations and releases resources before program termination.* 
 * @throws cv::Exception If OpenCV cleanup operations fail
 */
void cleanup();

} // namespace VideoStabilizer
