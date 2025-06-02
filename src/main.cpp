#include "stabilizer.hpp"
#include "camera_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // For timing measurements
#include <deque> // For frame buffer
#include <memory> // For std::unique_ptr

using namespace cv;
using namespace std;

enum class InputMode {
  UNSPECIFIED,
  SIMULATOR,
  CAMERA,
  FILE
};

int main(int argc, char* argv[]) {
  double fps = 0.0;
  InputMode chosenMode = InputMode::UNSPECIFIED;
  string videoFilePath;
  int cameraId = 0; // Default, will be validated if --camera is chosen
  string simulatorImagePath;

  CameraEngine::CameraParams default_camera_params;
  default_camera_params.position = Point3d(0.5, -0.3, 0.7);
  default_camera_params.pan = 0.0;
  default_camera_params.tilt = 180.0;
  default_camera_params.roll = 180.0;
  default_camera_params.focalLength = 1000.0;
  default_camera_params.sensorResolution = Size(1280, 720);

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
    cerr << "Error: No input mode specified. Use --simulator <path>, --camera <id>, or --file <path>." << endl;
    return -1;
  }
  if (totalModeFlags > 1) {
    cerr << "Error: Multiple input modes specified. Use only one of --simulator, --camera, or --file." << endl;
    return -1;
  }

  // Determine the chosen mode based on counts
  if (simulatorModeCount == 1) {
    chosenMode = InputMode::SIMULATOR;
  } else if (cameraModeCount == 1) {
    chosenMode = InputMode::CAMERA;
  } else if (fileModeCount == 1) {
    chosenMode = InputMode::FILE;
  } // No else needed due to prior checks

  // --- Second Pass: Parse Values for the Determined Mode and Handle Unknowns ---
  bool valueForChosenModeFound = false;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--simulator") {
      if (chosenMode == InputMode::SIMULATOR && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          simulatorImagePath = argv[++i]; // Consume value
          valueForChosenModeFound = true;
        } else {
          cerr << "Error: --simulator argument requires a file path." << endl;
          return -1;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --simulator flag encountered: " << arg << endl;
        return -1;
      }
    } else if (arg == "--camera") {
      if (chosenMode == InputMode::CAMERA && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          string cameraId_str = argv[++i]; // Consume ID string
          try {
            cameraId = stoi(cameraId_str);
            valueForChosenModeFound = true;
          } catch (const std::invalid_argument& ia) {
            cerr << "Error: Invalid camera ID for --camera: " << cameraId_str << endl;
            return -1;
          } catch (const std::out_of_range& oor) {
            cerr << "Error: Camera ID out of range for --camera: " << cameraId_str << endl;
            return -1;
          }
        } else {
          cerr << "Error: --camera argument requires a camera ID." << endl;
          return -1;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --camera flag encountered: " << arg << endl;
        return -1;
      }
    } else if (arg == "--file") {
      if (chosenMode == InputMode::FILE && !valueForChosenModeFound) {
        if (i + 1 < argc) {
          videoFilePath = argv[++i]; // Consume value
          valueForChosenModeFound = true;
        } else {
          cerr << "Error: --file argument requires a file path." << endl;
          return -1;
        }
      } else {
        cerr << "Error: Misplaced or duplicate --file flag encountered: " << arg << endl;
        return -1;
      }
    } else {
      // This argument is not "--simulator", "--camera", or "--file".
      // It's an unknown argument.
      cerr << "Error: Unknown argument: " << arg << endl;
      return -1;
    }
  }

  // Validate that the required value for the chosen mode was actually found and set
  if (chosenMode == InputMode::SIMULATOR && simulatorImagePath.empty()) {
    cerr << "Error: File path for --simulator was not successfully parsed or provided after the flag." << endl;
    return -1;
  }
  if (chosenMode == InputMode::FILE && videoFilePath.empty()) {
    cerr << "Error: File path for --file was not successfully parsed or provided after the flag." << endl;
    return -1;
  }
  if (chosenMode == InputMode::CAMERA && !valueForChosenModeFound) {
    cerr << "Error: Camera ID for --camera was not successfully parsed or provided after the flag." << endl;
    return -1;
  }

  // Initialize CameraEngine or VideoCapture based on source
  std::unique_ptr<CameraEngine> cameraEngine;
  VideoCapture cap;

  if (chosenMode == InputMode::CAMERA) {
    cap.open(cameraId);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!cap.isOpened()) {
      cout << "Error: Could not open camera with ID: " << cameraId << endl;
      return -1;
    }
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<double>(cap.get(CAP_PROP_FPS));
    if (fps <= 0) { // Webcam might return 0 fps
        fps = 30.0; // Default to 30 fps for webcams
        cout << "Warning: Camera FPS is 0, defaulting to " << fps << endl;
    }
    std::cout << "Using camera source. Frame width: " << frameWidth << ", Frame height: " << frameHeight << ", FPS: " << fps << std::endl;
  } else if (chosenMode == InputMode::FILE) {
    cap.open(videoFilePath);
    if (!cap.isOpened()) {
      cout << "Error: Could not open video file: " << videoFilePath << endl;
      return -1;
    }
    // Get video properties
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<double>(cap.get(CAP_PROP_FPS));
    if (fps <= 0) { // Some video files might return 0 fps
        fps = 30.0; // Default to 30 fps
        cout << "Warning: Video file FPS is 0 or invalid, defaulting to " << fps << endl;
    }

    std::cout << "Using file source: " << videoFilePath << std::endl;
    std::cout << "Frame width: " << frameWidth << std::endl;
    std::cout << "Frame height: " << frameHeight << std::endl;
    std::cout << "FPS: " << fps << std::endl;

  } else { // chosenMode == InputMode::SIMULATOR
    // Create the camera engine with floor texture
    cameraEngine = std::make_unique<CameraEngine>(simulatorImagePath, 10.0); // Construct only if simulator mode
    if (!cameraEngine) { // Should ideally not happen if make_unique succeeds
        cerr << "Error: Failed to create CameraEngine for simulator mode." << endl;
        return -1;
    }
    cameraEngine->setCameraParams(default_camera_params);
    fps = 30.0; // Simulated camera framerate
    std::cout << "Using simulator source with image: " << simulatorImagePath << ". FPS: " << fps << std::endl;
  }
  
  // Create stabilizer
  
  const int past_frames = 2.0 * fps;
  const int future_frames = 1.5 * fps;
  const int working_height = 360;
  Stabilizer stabilizer(past_frames, future_frames, working_height);
  
  // Create windows
  namedWindow("Original Camera Feed", WINDOW_NORMAL);
  namedWindow("Stabilized Output", WINDOW_NORMAL);
  
  cout << "Windows created. Using " << cv::getNumThreads() << " threads for rendering." << endl;
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

  // Buffer to store original frames for delay matching
  std::deque<cv::Mat> originalFrameBuffer;

  // --- Main Interaction Loop ---
  bool should_quit = false;
  while (!should_quit) {
    // Start timer for FPS calculation
    auto start = chrono::high_resolution_clock::now();

    // --- Process Keyboard Input ---
    int key = waitKey(1); // Wait 1ms for a key press

    if (key == 27) { // ESC key
      cout << "ESC pressed, exiting." << endl;
      break;
    }

    cv::Mat frame;

    if (chosenMode == InputMode::CAMERA || chosenMode == InputMode::FILE) {
      // Read frame from video file
      
      cap >> frame;
      if (frame.empty()) {
        if (chosenMode == InputMode::FILE) {
          cout << "End of video file reached or cannot read frame." << endl;
        } else {
          cerr << "Error: Could not read frame from camera." << endl;
        }
        break;
      }
    } else { // chosenMode == InputMode::SIMULATOR
      // Render frame from camera engine   
      if (!cameraEngine) {
          cerr << "Error: CameraEngine not initialized." << endl;
          break; 
      }
      frame = cameraEngine->renderFrame();   
      bool cameraMoved = false;
      switch (toupper(key)) {
        // --- Camera Movement ---
        case 'W': // Move Forward
          cameraEngine->moveForward(1.0);
          cameraMoved = true;
          break;
        case 'S': // Move Backward
          cameraEngine->moveBackward(1.0);
          cameraMoved = true;
          break;
        case 'A': // Move Left
          cameraEngine->moveLeft(1.0);
          cameraMoved = true;
          break;
        case 'D': // Move Right
          cameraEngine->moveRight(1.0);
          cameraMoved = true;
          break;
        case 'Q': // Roll Counter-Clockwise
          cameraEngine->rollCounterClockwise(10.0);
          cameraMoved = true;
          break;
        case 'E': // Roll Clockwise
          cameraEngine->rollClockwise(1.0);
          cameraMoved = true;
          break;
        case 32: // spacebar - Move Up
          cameraEngine->moveUp(1.0);
          cameraMoved = true;
          break;
        case 'C': // Move Down
          cameraEngine->moveDown(1.0);
          cameraMoved = true;
          break;
        case 'P': // Reset Camera Pose
          {
              cameraEngine->setCameraParams(default_camera_params);
              cameraMoved = true; // Indicate camera has changed
              cout << "Camera pose reset." << endl;
          }
          break;
      }
    }
    switch (toupper(key)) {
        
      // --- Stabilization Controls ---
      case 'X': // Reset stabilizer
        stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
        break;
      case 'F': // Full lock stabilization
        stabilizer.setStabilizationMode(StabilizationMode::ACCUMULATED_FULL_LOCK);
        break;
      case 'O': // ORB-based Full lock stabilization
        stabilizer.setStabilizationMode(StabilizationMode::ORB_FULL_LOCK);
        break;
      case 'L': // SIFT-based Full lock stabilization
        stabilizer.setStabilizationMode(StabilizationMode::SIFT_FULL_LOCK);
        break;
      case 'T': // Translation lock stabilization
        stabilizer.setStabilizationMode(StabilizationMode::TRANSLATION_LOCK);
        break;
      case 'R': // Rotation lock stabilization
        stabilizer.setStabilizationMode(StabilizationMode::ROTATION_LOCK);
        break;
      case 'G': // Global smoothing stabilization
        stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
        break;
    }

    
    // --- Apply Stabilization ---
    cv::Mat stabilized = frame;
    stabilized = stabilizer.stabilizeFrame(frame);
    
    // Store original frame in buffer
    originalFrameBuffer.push_back(frame.clone());
    
    // Display frames when we have enough to sync with the stabilizer's delay
    if (originalFrameBuffer.size() > future_frames) {
      // Get the delayed frame that matches the current stabilized frame
      Mat delayedOriginal = originalFrameBuffer.front();
      originalFrameBuffer.pop_front();
      
      if (chosenMode == InputMode::SIMULATOR) {
        if (!cameraEngine) {
             cerr << "Error: CameraEngine not available for displaying params in simulator mode." << endl;
        } else {
            // Get camera parameters to display
            CameraEngine::CameraParams& cameraParams = cameraEngine->getCameraParams();
            
            // Display information on original frame
            string posText = "Pos: (" + to_string(cameraParams.position.x).substr(0,4) + ", "
                            + to_string(cameraParams.position.y).substr(0,4) + ", "
                            + to_string(cameraParams.position.z).substr(0,4) + ")";
            // Add black background rectangle first
            Rect posRect = Rect(5, 10, 240, 25);
            rectangle(delayedOriginal, posRect, Scalar(0, 0, 0), -1);
            putText(delayedOriginal, posText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
            
            string rotText = "Pan:" + to_string(static_cast<int>(cameraParams.pan))
                           + " Tilt:" + to_string(static_cast<int>(cameraParams.tilt))
                           + " Roll:" + to_string(static_cast<int>(cameraParams.roll));
            // Add black background rectangle for rotation info
            Rect rotRect = Rect(5, 40, 240, 25);
            rectangle(delayedOriginal, rotRect, Scalar(0, 0, 0), -1);
            putText(delayedOriginal, rotText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
        }
      }
      
      // Stop timer and calculate FPS
      auto stop = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
      // Avoid division by zero if duration is very small
      double fps = (duration.count() > 0) ? 1000.0 / duration.count() : 2000.0;
      
      // Display FPS with black background
      string fpsText = "FPS: " + to_string(static_cast<int>(fps));
      Rect fpsRect = Rect(5, 70, 120, 25);
      rectangle(delayedOriginal, fpsRect, Scalar(0, 0, 0), -1);
      putText(delayedOriginal, fpsText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
      
      // Display frames
      imshow("Original Camera Feed", delayedOriginal);
      imshow("Stabilized Output", stabilized);
    } else {
      // We're still filling the buffer, show processing status
      cout << "Buffering frames: " << originalFrameBuffer.size() 
           << "/" << future_frames + 1 << "\r" << flush;
    }
  }

  // Release resources
  destroyAllWindows();
  cout << "Windows closed. Application finished." << endl;

  return 0;
}
