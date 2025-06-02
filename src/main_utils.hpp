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
  InputMode mode = InputMode::UNSPECIFIED;
  std::string path;  // Unified path field for video files or simulator images
  int cameraId = 0;
};

// Command line argument parsing
bool parseCommandLineArgs(int argc, char* argv[], InputConfig& config);

// Input source initialization
bool initializeInputSource(const InputConfig& config, double& fps, 
                          std::shared_ptr<CameraEngine> cameraEngine, 
                          cv::VideoCapture& cap, 
                          const CameraEngine::CameraParams& default_camera_params);

// Stabilizer and UI setup
Stabilizer setupStabilizerAndWindows(int past_frames, int future_frames, 
                                    int working_height);

// Input handling
bool handleCameraMovement(int key, std::shared_ptr<CameraEngine> cameraEngine, 
                         const CameraEngine::CameraParams& default_camera_params);
void handleStabilizationControls(int key, Stabilizer& stabilizer);

// Frame processing
bool captureFrame(const InputConfig& config, cv::VideoCapture& cap,
                 std::shared_ptr<CameraEngine> cameraEngine, cv::Mat& frame);
void addFrameOverlays(cv::Mat& frame, const InputConfig& config, 
                     std::shared_ptr<CameraEngine> cameraEngine, double fps);
void processAndDisplayFrames(cv::Mat frame, Stabilizer& stabilizer, 
                           std::deque<cv::Mat>& originalFrameBuffer, 
                           int future_frames, const InputConfig& config, 
                           std::shared_ptr<CameraEngine> cameraEngine, 
                           std::chrono::high_resolution_clock::time_point start_time);

// Cleanup
void cleanup();

} // namespace VideoStabilizer
