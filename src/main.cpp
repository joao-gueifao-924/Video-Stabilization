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

static const int ESC_KEY = 27;



// Video Stabilizer: Program Overview
//
// VIDEO INPUT TYPES:
// The program supports three distinct video input modes, specified via command line 
// arguments:
// 
// 1. CAMERA MODE: Live webcam input
//    - Captures real-time video from a connected camera/webcam
//    - Specified by camera ID parameter in command line arguments
//    - Demonstrates real-time video stabilization applications
// 
// 2. FILE MODE: Pre-recorded video file input  
//    - Processes existing video files (MP4, AVI, MOV, etc.)
//    - Specified by providing video file path in command line arguments
//    - Demonstrates stabilizing recorded footage
// 
// 3. SIMULATOR MODE: Synthetic camera simulation
//    - Generates artificial camera movement over a floor texture
//    - Creates precise, controllable camera motion for testing and demonstration
//    - Specified by simulator flag in command line arguments
//    - Allows keyboard control of camera position and orientation during runtime
// 
// KEYBOARD CONTROLS:
// The program supports various keyboard inputs for runtime control:
// 
//    UNIVERSAL CONTROLS (All input modes):
//       - ESC: Exit the program gracefully
//    
//    CAMERA MOVEMENT (Simulator mode only):
//       - W: Move camera forward
//       - A: Move camera left (strafe)
//       - S: Move camera backward  
//       - D: Move camera right (strafe)
//       - Q: Roll counter-clockwise
//       - E: Roll clockwise
//       - SPACE: Move camera up
//       - C: Move camera down
//       - P: Reset camera pose
//    
//    STABILIZATION CONTROLS (All input modes):
//       - X: Reset stabilizer (to Global Smoothing)
//       - F: Full lock stabilization (accumulated)
//       - O: ORB-based Full lock stabilization
//       - L: SIFT-based Full lock stabilization
//       - T: Translation lock stabilization mode
//       - R: Rotation lock stabilization mode
//       - G: Global smoothing stabilization mode
// 
// PROGRAM EXECUTION FLOW:
// 1. Parse command line arguments to determine input mode and configuration options
// 2. Initialize input source based on selected mode:
//    - Camera: Opens webcam with specified device ID
//    - File: Opens video file from provided path
//    - Simulator: Creates CameraEngine with synthetic floor texture and default 
//    camera parameters
// 3. Setup video stabilizer with configured parameters:
//    - Past window size (in frames) - how many previous frames to consider
//    - Future window size (in frames) - how many future frames to look ahead
//    - Working height for processing - resolution for stabilization calculations
// 4. Create display windows for original and stabilized video comparison
// 5. Main processing loop:
//    - Capture frame from selected input source
//    - Process frame through stabilization algorithm
//    - Display both original and stabilized frames side-by-side
//    - Handle keyboard input for runtime controls (camera movement in simulator mode, 
//    stabilization controls, ESC to exit)
//    - Continue processing until ESC key is pressed or input source ends
// 6. Cleanup resources and exit gracefully


static const CameraEngine::CameraParams default_camera_params = {
  Point3d(0.5, -0.3, 0.7),
  0.0,
  180.0,
  180.0,
  1000.0,
  Size(1280, 720)
};

int main(int argc, char* argv[]) {
  double fps = 0.0;
  InputConfig config;

  if (!parseCommandLineArgs(argc, argv, config)) {
    return EXIT_FAILURE;
  }

  std::shared_ptr<CameraEngine> cameraEngine;
  VideoCapture cap;
  
  if (!initializeInputSource(config, fps, cameraEngine, cap, 
                            default_camera_params)) {
    return EXIT_FAILURE;
  }
  
  const int past_frames    = config.pastWindowSecs * fps;
  const int future_frames  = config.futureWindowSecs * fps;
  const int working_height = config.workingHeight;
  Stabilizer stabilizer = setupStabilizerAndWindows(past_frames, future_frames, 
                                                    working_height);
  
  // Buffer to store original frames for delayed matching with stabilized frames
  // This is used to present both the original and stabilized frames in sync
  std::deque<cv::Mat> originalFrameBuffer;

  // --- Main Interaction Loop ---
  bool should_quit = false;
  while (!should_quit) {
    // Start timer for FPS calculation
    auto start = chrono::high_resolution_clock::now();

    int key = waitKey(1);

    if (key == ESC_KEY) {
      cout << "ESC pressed, exiting." << endl;
      break;
    }

    // Handle keyboard input for simulator mode BEFORE capturing frame
    if (config.mode == InputMode::SIMULATOR) {
      handleCameraMovement(key, cameraEngine, default_camera_params);
    }
    
    cv::Mat frame;
    
    if (!captureFrame(config, cap, cameraEngine, frame)) {
      break;
    }
    
    handleStabilizationControls(key, stabilizer);
    
    processAndDisplayFrames(frame, stabilizer, originalFrameBuffer, 
                          future_frames, config, cameraEngine, start);
  }

  cleanup();
  return EXIT_SUCCESS;
}
