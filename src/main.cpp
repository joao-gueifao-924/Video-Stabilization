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

static const std::pair<double, double> VIDEO_STABILIZER_WINDOW_SECS = {2.0, 1.5};
static const int VIDEO_STABILIZER_WORKING_HEIGHT_PIXELS = 360;

static const int ESC_KEY = 27;

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
  
  const int past_frames    = VIDEO_STABILIZER_WINDOW_SECS.first * fps;
  const int future_frames  = VIDEO_STABILIZER_WINDOW_SECS.second * fps;
  const int working_height = VIDEO_STABILIZER_WORKING_HEIGHT_PIXELS;
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
