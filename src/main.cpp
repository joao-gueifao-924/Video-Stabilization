#include "stabilizer.hpp"
#include "camera_engine.hpp"
#include "main_utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <deque>
#include <memory>

using namespace cv;
using namespace std;
using namespace VideoStabilizer;

// ASCII code for the ESC key, used for graceful program termination.
static const int ESC_KEY = 27;

/**
 * Default camera parameters used for simulator mode initialization.
 * 
 * @note These parameters define the initial camera state in the 3D simulated environment:
 *       - position: 3D coordinates in world space
 *       - pan, tilt, roll: Camera rotation angles in degrees
 *       - focalLength: Camera focal length in pixels (controls field of view)
 *       - resolution: Output frame dimensions
 * 
 * @see CameraEngine::CameraParams for detailed parameter descriptions
 * @see handleCameraMovement() for runtime camera control in simulator mode
 */
static const CameraEngine::CameraParams default_camera_params = {
  Point3d(0.5, -0.3, 0.7),
  0.0,
  180.0,
  180.0,
  1000.0,
  Size(1280, 720)
};

/**
 * Main entry point for the Video Stabilization application.
 * 
 * @param argc Number of command line arguments passed to the program
 * @param argv Array of command line argument strings
 * 
 * @return EXIT_SUCCESS (0) on successful completion, EXIT_FAILURE (1) on error
 * 
 * @section video_stabilization VIDEO STABILIZATION
 * Video stabilization is a technique that reduces unwanted camera motion in video footage,
 * making the output smoother. This program specifically focuses on real-time video
 * stabilization, meaning it can process and stabilize video as it's being captured,
 * rather than requiring post-processing.
 * 
 * The program offers several stabilization modes:
 * 
 * 1. **Global Smoothing Mode**: Applies smoothing to reduce overall camera shake while
 *    maintaining natural motion.
 * 
 * 2. **Lock Modes**:
 *    - **Full Lock**: Completely cancels out camera motion, keeping the scene perfectly still
 *    - **Translation Lock**: Only stabilizes translational movement (side-to-side, up-down)
 *    - **Rotation Lock**: Only stabilizes rotational movement (camera roll)
 * 
 * The program supports three input modes:
 * - Live webcam input for real-time stabilization
 * - Pre-recorded video file processing
 * - A simulator mode that generates synthetic camera movement for testing
 * 
 * This implementation offers multiple frame registration methods for the motion locking
 * modes listed above:
 * - Accumulated Motion with Sparse Optical Flow
 * - ORB-based Frame Registration
 * - SIFT-based Frame Registration
 * 
 * This allows users to compare different stabilization approaches and choose the one that
 * works best for their specific use case. The program can be controlled in real-time using
 * keyboard shortcuts to switch between different stabilization modes and adjust parameters.
 * 
 * The real-time nature of this implementation makes it particularly useful for applications
 * like live streaming, video conferencing, or any scenario where immediate stabilization
 * is required rather than post-processing.
 * 
 * @section video_input_types VIDEO INPUT TYPES
 * The program supports three distinct video input modes, specified via command line 
 * arguments:
 * 
 * 1. **CAMERA MODE**: Live webcam input
 *    - Captures real-time video from a connected camera/webcam
 *    - Specified by camera ID parameter in command line arguments
 *    - Demonstrates real-time video stabilization applications
 * 
 * 2. **FILE MODE**: Pre-recorded video file input  
 *    - Processes existing video files (MP4, AVI, MOV, etc.)
 *    - Specified by providing video file path in command line arguments
 *    - Demonstrates stabilizing recorded footage
 * 
 * 3. **SIMULATOR MODE**: Synthetic camera simulation
 *    - Generates artificial camera movement over a floor texture
 *    - Creates precise, controllable camera motion for testing and demonstration
 *    - Specified by simulator flag in command line arguments
 *    - Allows keyboard control of camera position and orientation during runtime
 *
 * @section motion_tracking_comparison MOTION TRACKING COMPARISON
 * The program provides a unique capability to compare different frame registration methods in the 
 * context of real-time camera motion cancellation (vision locking).
 * 
 * - **Accumulated Motion with Sparse Optical Flow**: Computes sparse optical flow between 
 *   consecutive frames, then fits an invertible 3x3 rigid body transformation matrix to this flow, 
 *   and finally aggregates these transformations through chained matrix multiplication.
 * - **ORB-based Frame Registration**: Uses Oriented FAST and Rotated BRIEF (ORB) feature detection 
 *   and matching for efficient feature-based frame registration. A rigid body transformation matrix is 
 *   fitted to the feature matches.
 * - **SIFT-based Frame Registration**: Employs Scale-Invariant Feature Transform (SIFT) for robust
 *   feature-based frame registration. A rigid body transformation matrix is fitted to the feature matches.
 * 
 * This comparison functionality allows users to evaluate the performance and
 * characteristics of each frame registration method under different conditions and motion
 * patterns.
 *
 * Refer to @ref VideoStabilizer::printUsage() for more details on how to construct the 
 * command line arguments.
 * 
 * @section keyboard_controls KEYBOARD CONTROLS
 * The program supports various keyboard inputs for runtime control:
 * 
 *    **UNIVERSAL CONTROLS** (All input modes):
 *       - ESC: Exit the program gracefully
 *    
 *    **CAMERA MOVEMENT** (Simulator mode only):
 *       - W: Move camera forward
 *       - A: Move camera left (strafe)
 *       - S: Move camera backward  
 *       - D: Move camera right (strafe)
 *       - Q: Roll counter-clockwise
 *       - E: Roll clockwise
 *       - SPACE: Move camera up
 *       - C: Move camera down
 *       - P: Reset camera pose
 *    
 *    **STABILIZATION CONTROLS** (All input modes):
 *       - X: Reset stabilizer (to Global Smoothing)
 *       - F: Full lock stabilization (accumulated)
 *       - O: ORB-based Full lock stabilization
 *       - L: SIFT-based Full lock stabilization
 *       - T: Translation lock stabilization mode
 *       - R: Rotation lock stabilization mode
 *       - G: Global smoothing stabilization mode
 * 
 * @section program_execution_flow PROGRAM EXECUTION FLOW
 * 1. Parse command line arguments to determine input mode and configuration options
 * 2. Initialize input source based on selected mode:
 *    - Camera: Opens webcam with specified device ID
 *    - File: Opens video file from provided path
 *    - Simulator: Creates CameraEngine with synthetic floor texture and default 
 *      camera parameters
 * 3. Setup video stabilizer with configured parameters:
 *    - Past window size (in frames) - how many previous frames to consider
 *    - Future window size (in frames) - how many future frames to look ahead
 *    - Working height for processing - resolution for stabilization calculations
 * 4. Create display windows for original and stabilized video comparison
 * 5. Main processing loop:
 *    - Capture frame from selected input source
 *    - Process frame through stabilization algorithm
 *    - Display both original and stabilized frames side-by-side
 *    - Handle keyboard input for runtime controls (camera movement in simulator mode, 
 *      stabilization controls, ESC to exit)
 *    - Continue processing until ESC key is pressed or input source ends
 * 6. Cleanup resources and exit gracefully
 * 
 * @note The program creates two OpenCV windows:
 *       - "Original": Shows unstabilized input frames
 *       - "Stabilized": Shows processed stabilized frames
 * 
 * @note Frame processing includes a buffer system to synchronize display
 *       of original and stabilized frames, accounting for the stabilizer's
 *       future frame window delay.
 * 
 * @throws multiple exceptions. As this is a demonstration program, we deliberately 
 *         allow exceptions to propagate and crash the program to highlight issues.
 * 
 * @see VideoStabilizer::parseCommandLineArgs() for command line argument details
 * @see VideoStabilizer::initializeInputSource() for input source initialization
 * @see VideoStabilizer::setupStabilizerAndWindows() for stabilizer configuration
 */
int main(int argc, char* argv[]) {
  double fps = 0.0;  // Frames per second of the input source
  InputConfig config;  // Configuration parsed from command line arguments

  // Parse command line arguments and validate configuration
  if (!parseCommandLineArgs(argc, argv, config)) {
    return EXIT_FAILURE;
  }

  // Initialize input source objects based on selected mode
  std::shared_ptr<CameraEngine> cameraEngine;  // For simulator mode
  VideoCapture cap;  // For camera and file modes
  
  // Setup the appropriate input source (camera, file, or simulator)
  if (!initializeInputSource(config, fps, cameraEngine, cap, 
                            default_camera_params)) {
    return EXIT_FAILURE;
  }
  
  // Calculate stabilizer window parameters based on FPS and configuration
  const int past_frames    = config.pastWindowSecs * fps;    // Historical frames for smoothing
  const int future_frames  = config.futureWindowSecs * fps;  // Lookahead frames for smoothing
  const int working_height = config.workingHeight;           // Processing resolution height
  
  // Initialize stabilizer with calculated parameters and create display windows
  Stabilizer stabilizer = setupStabilizerAndWindows(past_frames, future_frames, 
                                                    working_height);
  
  // Buffer to store original frames for delayed matching with stabilized frames.
  // Ensures original and stabilized frames are displayed in temporal alignment,
  // accounting for the stabilizer's processing delay from future frame window.
  // Buffer size auto-adjusts based on stabilizer's future window size.
  std::deque<cv::Mat> originalFrameBuffer;

  // --- Main Interaction Loop ---
  // Process frames continuously until user requests exit or input source ends
  while (true) {
    // Start high-resolution timer for accurate FPS calculation
    auto start = chrono::high_resolution_clock::now();

    // Check for keyboard input (non-blocking, 1ms timeout)
    int key = waitKey(1);

    // Handle ESC key for graceful program termination
    if (key == ESC_KEY) {
      cout << "ESC pressed, exiting." << endl;
      break;
    }

    // Process camera movement controls for simulator mode.
    // Note: This is handled BEFORE frame capture to ensure movement is applied
    // to the current frame being rendered.
    if (config.mode == InputMode::SIMULATOR) {
      handleCameraMovement(key, cameraEngine, default_camera_params);
    }
    
    cv::Mat frame;  // Current frame from input source
    
    // Capture next frame from the configured input source
    if (!captureFrame(config, cap, cameraEngine, frame)) {
      // End of input reached or capture failed - exit loop
      break;
    }
    
    // Process stabilization control keyboard inputs
    handleStabilizationControls(key, stabilizer);
    
    // Process frame through stabilizer, manage display buffers, and update UI
    processAndDisplayFrames(frame, stabilizer, originalFrameBuffer, 
                          future_frames, config, cameraEngine, start);
  }

  // Perform cleanup operations and release all resources
  cleanup();
  return EXIT_SUCCESS;
}
