#include "main_utils.hpp"
#include <iostream>

using namespace cv;
using namespace std;

namespace VideoStabilizer {

// Minimum window size in seconds to ensure correct stabilizer runtime
static const double MIN_STABILIZER_WINDOW_SECONDS = 0.030; // 30 milliseconds minimum

void printUsage(const char* programName) {
  cout << "Usage: " << programName << " <input_mode> [options]" << endl;
  cout << endl;
  cout << "Input modes (required, choose one):" << endl;
  cout << "  --simulator <path>    Use simulator with floor texture image" << endl;
  cout << "  --camera <id>         Use camera with given ID (typically 0)" << endl;
  cout << "  --file <path>         Use video file" << endl;
  cout << endl;
  cout << "Optional stabilizer parameters:" << endl;
  cout << "  --past-window <secs>     Past window size in seconds (default: 2.0)" << endl;
  cout << "  --future-window <secs>   Future window size in seconds (default: 1.5)" << endl;
  cout << "  --working-height <pixels> Working height in pixels (default: 360)" << endl;
  cout << "                           Must be > 90 and <= 2160" << endl;
  cout << endl;
  cout << "Note: Total window size (--past-window + --future-window) must be >= " 
       << MIN_STABILIZER_WINDOW_SECONDS << " seconds" << endl;
  cout << endl;
  cout << "Examples:" << endl;
  cout << "  " << programName << " --camera 0" << endl;
  cout << "  " << programName << " --file video.mp4 --past-window 3.0 --future-window 2.0" << endl;  
  cout << "  " << programName << " --simulator texture.jpg --working-height 480" << endl;
}

bool parseCommandLineArgs(int argc, char* argv[], InputConfig& config) {
  // Check for help request
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return false; // Not an error, but we don't want to continue
    }
  }

  // --- First Pass: Identify Input Mode and Count --- 
  int simulatorModeCount = 0;
  int cameraModeCount = 0;
  int fileModeCount = 0;

  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--simulator") {
      simulatorModeCount++;
    } else if (arg == "--camera") {
      cameraModeCount++;
    } else if (arg == "--file") {
      fileModeCount++;
    } // In this pass, we only count mode flags, not their values or other args
  }

  int totalModeFlags = simulatorModeCount + cameraModeCount + fileModeCount;

  if (totalModeFlags == 0) {
    cerr << "Error: No input mode specified." << endl;
    printUsage(argv[0]);
    return false;
  }
  if (totalModeFlags > 1) {
    cerr << "Error: Multiple input modes specified. Use only one of "
         << "--simulator, --camera, or --file." << endl;
    return false;
  }

  // Determine the chosen mode based on counts
  if (simulatorModeCount == 1) {
    config.mode = InputMode::SIMULATOR;
  } else if (cameraModeCount == 1) {
    config.mode = InputMode::CAMERA;
  } else if (fileModeCount == 1) {
    config.mode = InputMode::FILE;
  } // No else needed due to prior checks

  // --- Second Pass: Parse Values for the Determined Mode and Handle Optional Args ---
  bool valueForChosenModeFound = false;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--simulator") {
      if (config.mode == InputMode::SIMULATOR && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          config.path = argv[++i]; // Consume value
          valueForChosenModeFound = true;
        } else {
          cerr << "Error: --simulator argument requires a file path." << endl;
          return false;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --simulator flag encountered: "
             << arg << endl;
        return false;
      }
    } else if (arg == "--camera") {
      if (config.mode == InputMode::CAMERA && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          string cameraId_str = argv[++i]; // Consume ID string
          try {
            config.cameraId = stoi(cameraId_str);
            valueForChosenModeFound = true;
          } catch (const std::invalid_argument& ia) {
            cerr << "Error: Invalid camera ID for --camera: " 
                 << cameraId_str << endl;
            return false;
          } catch (const std::out_of_range& oor) {
            cerr << "Error: Camera ID out of range for --camera: " 
                 << cameraId_str << endl;
            return false;
          }
        } else {
          cerr << "Error: --camera argument requires a camera ID." << endl;
          return false;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --camera flag encountered: "
             << arg << endl;
        return false;
      }
    } else if (arg == "--file") {
      if (config.mode == InputMode::FILE && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          config.path = argv[++i]; // Consume value
          valueForChosenModeFound = true;
        } else {
          cerr << "Error: --file argument requires a file path." << endl;
          return false;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --file flag encountered: "
             << arg << endl;
        return false;
      }
    } else if (arg == "--past-window") {
      if (i + 1 < argc) {
        try {
          config.pastWindowSecs = stod(argv[++i]);
          if (config.pastWindowSecs < 0) {
            cerr << "Error: --past-window must be non-negative." << endl;
            return false;
          }
        } catch (const std::invalid_argument& ia) {
          cerr << "Error: Invalid value for --past-window: " << argv[i] << endl;
          return false;
        } catch (const std::out_of_range& oor) {
          cerr << "Error: Value out of range for --past-window: " << argv[i] << endl;
          return false;
        }
      } else {
        cerr << "Error: --past-window argument requires a value in seconds." << endl;
        return false;
      }
    } else if (arg == "--future-window") {
      if (i + 1 < argc) {
        try {
          config.futureWindowSecs = stod(argv[++i]);
          if (config.futureWindowSecs < 0) {
            cerr << "Error: --future-window must be non-negative." << endl;
            return false;
          }
        } catch (const std::invalid_argument& ia) {
          cerr << "Error: Invalid value for --future-window: " << argv[i] << endl;
          return false;
        } catch (const std::out_of_range& oor) {
          cerr << "Error: Value out of range for --future-window: " << argv[i] << endl;
          return false;
        }
      } else {
        cerr << "Error: --future-window argument requires a value in seconds." << endl;
        return false;
      }
    } else if (arg == "--working-height") {
      if (i + 1 < argc) {
        try {
          config.workingHeight = stoi(argv[++i]);
          if (config.workingHeight <= 90) {
            cerr << "Error: --working-height must be greater than 90 pixels." << endl;
            return false;
          }
          if (config.workingHeight > 2160) {
            cerr << "Error: --working-height must be no more than 2160 pixels." << endl;
            return false;
          }
        } catch (const std::invalid_argument& ia) {
          cerr << "Error: Invalid value for --working-height: " << argv[i] << endl;
          return false;
        } catch (const std::out_of_range& oor) {
          cerr << "Error: Value out of range for --working-height: " << argv[i] << endl;
          return false;
        }
      } else {
        cerr << "Error: --working-height argument requires a value in pixels." << endl;
        return false;
      }
    } else {
      // This argument is not a recognized flag.
      // It's an unknown argument.
      cerr << "Error: Unknown argument: " << arg << endl;
      return false;
    }
  }

  // Validate that both past and future windows are not both effectively zero
  // This is to ensure that the stabilizer can run correctly.
  const double totalWindowSize = config.pastWindowSecs + config.futureWindowSecs;
  if (totalWindowSize < MIN_STABILIZER_WINDOW_SECONDS) {
    cerr << "Error: Total window size must be >= " << MIN_STABILIZER_WINDOW_SECONDS << " seconds." << endl;
    cerr << "Adjust --past-window and/or --future-window to increase the total window size." << endl;
    return false;
  }

  // Validate that the required value for the chosen mode was actually found
  if (config.mode == InputMode::SIMULATOR && config.path.empty()) {
    cerr << "Error: File path for --simulator was not successfully parsed "
         << "or provided after the flag." << endl;
    return false;
  }
  if (config.mode == InputMode::FILE && config.path.empty()) {
    cerr << "Error: File path for --file was not successfully parsed "
         << "or provided after the flag." << endl;
    return false;
  }
  if (config.mode == InputMode::CAMERA && !valueForChosenModeFound) {
    cerr << "Error: Camera ID for --camera was not successfully parsed "
         << "or provided after the flag." << endl;
    return false;
  }

  return true;
}

bool initializeInputSource(const InputConfig& config, double& fps, 
                          std::shared_ptr<CameraEngine> cameraEngine, 
                          VideoCapture& cap, 
                          const CameraEngine::CameraParams& default_camera_params) {
  
  if (config.mode == InputMode::CAMERA) {
    cap.open(config.cameraId);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!cap.isOpened()) {
      cout << "Error: Could not open camera with ID: " << config.cameraId << endl;
      return false;
    }
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<double>(cap.get(CAP_PROP_FPS));
    if (fps <= 0) { // Webcam might return 0 fps
        fps = 30.0; // Default to 30 fps for webcams
        cout << "Warning: Camera FPS is 0, defaulting to " << fps << endl;
    }
    std::cout << "Using camera source. Frame width: " << frameWidth 
              << ", Frame height: " << frameHeight << ", FPS: " << fps 
              << std::endl;
    
  } else if (config.mode == InputMode::FILE) {
    cap.open(config.path);
    if (!cap.isOpened()) {
      cout << "Error: Could not open video file: " << config.path << endl;
      return false;
    }
    // Get video properties
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<double>(cap.get(CAP_PROP_FPS));
    if (fps <= 0) { // Some video files might return 0 fps
        fps = 30.0; // Default to 30 fps
        cout << "Warning: Video file FPS is 0 or invalid, defaulting to " 
             << fps << endl;
    }

    std::cout << "Using file source: " << config.path << std::endl;
    std::cout << "Frame width: " << frameWidth << std::endl;
    std::cout << "Frame height: " << frameHeight << std::endl;
    std::cout << "FPS: " << fps << std::endl;

  } else { // config.mode == InputMode::SIMULATOR
    // Create the camera engine with floor texture
    cameraEngine = std::make_shared<CameraEngine>(config.path, 10.0);
    if (!cameraEngine) {
        cerr << "Error: Failed to create CameraEngine for simulator mode." 
             << endl;
        return false;
    }
    cameraEngine->setCameraParams(default_camera_params);
    fps = 30.0; // Simulated camera framerate
    std::cout << "Using simulator source with image: " << config.path 
              << ". FPS: " << fps << std::endl;
  }
  
  return true;
}

Stabilizer setupStabilizerAndWindows(int past_frames, int future_frames, 
                                    int working_height) {
  Stabilizer stabilizer(past_frames, future_frames, working_height);
  
  // Create windows
  namedWindow("Original Camera Feed", WINDOW_NORMAL);
  namedWindow("Stabilized Output", WINDOW_NORMAL);
  
  cout << "\nControls:\n"
       << " W/S: Move Forward/Backward (relative to camera direction)\n"
       << " A/D: Move Left/Right (relative to camera direction)\n"
       << " Q/E: Roll Counter-Clockwise / Clockwise\n"
       << " Space: Move Up\n"
       << " C: Move Down\n"
       << " X: Reset stabilizer (to Global Smoothing)\n"
       << " F: Full lock stabilization (accumulated)\n"
       << " O: ORB-based Full lock stabilization\n"
       << " L: SIFT-based Full lock stabilization\n"
       << " T: Translation lock stabilization mode\n"
       << " R: Rotation lock stabilization mode\n"
       << " G: Global smoothing stabilization mode\n"
       << " P: Reset Camera Pose\n"
       << " ESC: Exit\n" << endl;
       
  return stabilizer;
}

bool handleCameraMovement(int key, std::shared_ptr<CameraEngine> cameraEngine, 
                         const CameraEngine::CameraParams& default_camera_params) {
  if (!cameraEngine) return false;
  
  bool has_camera_moved = true;
  const int SPACEBAR_KEY = 32;
  switch (toupper(key)) {
    case 'W':
      cameraEngine->moveForward(1.0);
      break;
    case 'S':
      cameraEngine->moveBackward(1.0);
      break;
    case 'A':
      cameraEngine->moveLeft(1.0);
      break;
    case 'D':
      cameraEngine->moveRight(1.0);
      break;
    case 'Q':
      cameraEngine->rollCounterClockwise(10.0);
      break;
    case 'E':
      cameraEngine->rollClockwise(1.0);
      break;
    case SPACEBAR_KEY:
      cameraEngine->moveUp(1.0);
      break;
    case 'C':
      cameraEngine->moveDown(1.0);
      break;
    case 'P':
      {
          cameraEngine->setCameraParams(default_camera_params);
          cout << "Camera pose reset." << endl;
      }
      break;
    default:
      has_camera_moved = false;
      break;
  }
  return has_camera_moved;
}

void handleStabilizationControls(int key, Stabilizer& stabilizer) {
  switch (toupper(key)) {
    case 'X':
      stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
      break;
    case 'F':
      stabilizer.setStabilizationMode(StabilizationMode::ACCUMULATED_FULL_LOCK);
      break;
    case 'O':
      stabilizer.setStabilizationMode(StabilizationMode::ORB_FULL_LOCK);
      break;
    case 'L':
      stabilizer.setStabilizationMode(StabilizationMode::SIFT_FULL_LOCK);
      break;
    case 'T':
      stabilizer.setStabilizationMode(StabilizationMode::TRANSLATION_LOCK);
      break;
    case 'R':
      stabilizer.setStabilizationMode(StabilizationMode::ROTATION_LOCK);
      break;
    case 'G':
      stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
      break;
  }
}

bool captureFrame(const InputConfig& config, VideoCapture& cap,
                 std::shared_ptr<CameraEngine> cameraEngine, cv::Mat& frame) {
  if (config.mode == InputMode::CAMERA || config.mode == InputMode::FILE) {
    cap >> frame;
    if (frame.empty()) {
      if (config.mode == InputMode::FILE) {
        cout << "End of video file reached or cannot read frame." << endl;
      } else {
        cerr << "Error: Could not read frame from camera." << endl;
      }
      return false;
    }
  } else { // config.mode == InputMode::SIMULATOR
    if (!cameraEngine) {
        cerr << "Error: CameraEngine not initialized." << endl;
        return false; 
    }
    frame = cameraEngine->renderFrame();   
  }
  return true;
}

void addFrameOverlays(cv::Mat& frame, const InputConfig& config, 
                     std::shared_ptr<CameraEngine> cameraEngine, double fps) {
  if (config.mode == InputMode::SIMULATOR) {
    if (!cameraEngine) {
         cerr << "Error: CameraEngine not available for displaying params "
              << "in simulator mode." << endl;
    } else {
        // Get camera parameters to display
        const CameraEngine::CameraParams& cameraParams = 
            cameraEngine->getCameraParams();
        
        // Display information on original frame
        string posText = "Pos: (" + to_string(cameraParams.position.x).substr(0,4) 
                        + ", " + to_string(cameraParams.position.y).substr(0,4) 
                        + ", " + to_string(cameraParams.position.z).substr(0,4) + ")";
        // Add black background rectangle first
        Rect posRect = Rect(5, 10, 240, 25);
        rectangle(frame, posRect, Scalar(0, 0, 0), -1);
        putText(frame, posText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, 
                Scalar(0, 255, 0), 1);
        
        string rotText = "Pan:" + to_string(static_cast<int>(cameraParams.pan))
                       + " Tilt:" + to_string(static_cast<int>(cameraParams.tilt))
                       + " Roll:" + to_string(static_cast<int>(cameraParams.roll));
        // Add black background rectangle for rotation info
        Rect rotRect = Rect(5, 40, 240, 25);
        rectangle(frame, rotRect, Scalar(0, 0, 0), -1);
        putText(frame, rotText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, 
                Scalar(0, 255, 0), 1);
    }
  }
  
  // Display FPS with black background
  string fpsText = "FPS: " + to_string(static_cast<int>(fps));
  Rect fpsRect = Rect(5, 70, 120, 25);
  rectangle(frame, fpsRect, Scalar(0, 0, 0), -1);
  putText(frame, fpsText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, 
          Scalar(0, 255, 0), 1);
}

void processAndDisplayFrames(cv::Mat frame, Stabilizer& stabilizer, 
                           std::deque<cv::Mat>& originalFrameBuffer, 
                           int future_frames, const InputConfig& config, 
                           std::shared_ptr<CameraEngine> cameraEngine, 
                           std::chrono::high_resolution_clock::time_point start_time) {
  // --- Apply Stabilization ---
  cv::Mat stabilized = stabilizer.stabilizeFrame(frame);
  
  // Store original frame in buffer
  originalFrameBuffer.push_back(frame.clone());
  
  // Display frames when we have enough to sync with the stabilizer's delay
  if (originalFrameBuffer.size() > future_frames) {
    // Get the delayed frame that matches the current stabilized frame
    Mat delayedOriginal = originalFrameBuffer.front();
    originalFrameBuffer.pop_front();
    
    // Stop timer and calculate FPS
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start_time);
    // Avoid division by zero if duration is very small
    double fps = (duration.count() > 0) ? 1000.0 / duration.count() : 2000.0;
    
    // Add overlays to the delayed original frame
    addFrameOverlays(delayedOriginal, config, cameraEngine, fps);
    
    // Display frames
    imshow("Original Camera Feed", delayedOriginal);
    imshow("Stabilized Output", stabilized);
  } else {
    // We're still filling the buffer, show processing status
    cout << "Buffering frames: " << originalFrameBuffer.size() 
         << "/" << future_frames + 1 << "\r" << flush;
  }
}

void cleanup() {
  destroyAllWindows();
  cout << "Windows closed. Application finished." << endl;
}

} // namespace VideoStabilizer
