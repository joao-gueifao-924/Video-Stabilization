#include "stabilizer.hpp"
#include "tracker.hpp"
#include "camera_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // For timing measurements
#include <deque> // For frame buffer

using namespace cv;
using namespace std;

int main() {
  // Create the camera engine with floor texture
  CameraEngine cameraEngine("/home/joao/Downloads/pexels-pixabay-326055.jpg", 10.0);
  
  // Create stabilizer
  const double fps = 30.0; // Simulated camera framerate
  const int past_frames = 0.5 * fps;
  const int future_frames = 0.1 * fps;
  const int working_height = 540;
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
       << " X: Reset stabilizer\n"
       << " F: Full lock stabilization mode\n"
       << " T: Translation lock stabilization mode\n"
       << " R: Rotation lock stabilization mode\n"
       << " G: Global smoothing stabilization mode\n"
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

    bool cameraMoved = false;
    switch (key) {
      // --- Camera Movement ---
      case 'w': // Move Forward
      case 'W':
        cameraEngine.moveForward(1.0);
        cameraMoved = true;
        break;
      case 's': // Move Backward
      case 'S':
        cameraEngine.moveBackward(1.0);
        cameraMoved = true;
        break;
      case 'a': // Move Left
      case 'A':
        cameraEngine.moveLeft(1.0);
        cameraMoved = true;
        break;
      case 'd': // Move Right
      case 'D':
        cameraEngine.moveRight(1.0);
        cameraMoved = true;
        break;
      case 'q': // Roll Counter-Clockwise
      case 'Q':
        cameraEngine.rollCounterClockwise(1.0);
        cameraMoved = true;
        break;
      case 'e': // Roll Clockwise
      case 'E':
        cameraEngine.rollClockwise(1.0);
        cameraMoved = true;
        break;
      case 32: // spacebar - Move Up
        cameraEngine.moveUp(1.0);
        cameraMoved = true;
        break;
      case 'c': // Move Down
      case 'C':
        cameraEngine.moveDown(1.0);
        cameraMoved = true;
        break;
        
      // --- Stabilization Controls ---
      case 'x': // Reset stabilizer
      case 'X':
        stabilizer.reset();
        originalFrameBuffer.clear();
        break;
      case 'f': // Full lock stabilization
      case 'F':
        stabilizer.setStabilizationMode(StabilizationMode::FULL_LOCK);
        break;
      case 't': // Translation lock stabilization
      case 'T':
        stabilizer.setStabilizationMode(StabilizationMode::TRANSLATION_LOCK);
        break;
      case 'r': // Rotation lock stabilization
      case 'R':
        stabilizer.setStabilizationMode(StabilizationMode::ROTATION_LOCK);
        break;
      case 'g': // Global smoothing stabilization
      case 'G':
        stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
        break;
    }

    // --- Render Camera Frame ---
    Mat frame = cameraEngine.renderFrame();
    
    // --- Apply Stabilization ---
    Mat stabilized = stabilizer.stabilizeFrame(frame);
    
    // Store original frame in buffer
    originalFrameBuffer.push_back(frame.clone());
    
    // Display frames when we have enough to sync with the stabilizer's delay
    if (originalFrameBuffer.size() > future_frames) {
      // Get the delayed frame that matches the current stabilized frame
      Mat delayedOriginal = originalFrameBuffer.front();
      originalFrameBuffer.pop_front();
      
      // Get camera parameters to display
      CameraEngine::CameraParams& cameraParams = cameraEngine.getCameraParams();
      
      // Display information on original frame
      string posText = "Pos: (" + to_string(cameraParams.position.x).substr(0,4) + ", "
                      + to_string(cameraParams.position.y).substr(0,4) + ", "
                      + to_string(cameraParams.position.z).substr(0,4) + ")";
      putText(delayedOriginal, posText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
      
      string rotText = "Pan:" + to_string(static_cast<int>(cameraParams.pan))
                     + " Tilt:" + to_string(static_cast<int>(cameraParams.tilt))
                     + " Roll:" + to_string(static_cast<int>(cameraParams.roll));
      putText(delayedOriginal, rotText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
      
      // Stop timer and calculate FPS
      auto stop = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
      // Avoid division by zero if duration is very small
      double fps = (duration.count() > 0) ? 1000.0 / duration.count() : 2000.0;
      
      // Display FPS
      string fpsText = "FPS: " + to_string(static_cast<int>(fps));
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
