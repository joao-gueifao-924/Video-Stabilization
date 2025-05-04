#include "stabilizer.hpp"
#include "tracker.hpp"
#include "camera_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // For timing measurements
#include <deque> // For frame buffer

using namespace cv;
using namespace std;


int main2() {
  // Create the camera engine with floor texture
  CameraEngine cameraEngine("/home/joao/Downloads/pexels-pixabay-326055.jpg", 10.0);
  
  // Create a window
  namedWindow("Rendered Frame - Interactive Control", WINDOW_NORMAL);
  cout << "Window created. Using " << cv::getNumThreads() << " threads for rendering." << endl;
  cout << "\nControls:\n"
       << " W/S: Move Forward/Backward (relative to camera direction)\n"
       << " A/D: Move Left/Right (relative to camera direction)\n"
       << " Q/E: Roll Counter-Clockwise / Clockwise\n"
       << " Space: Move Up\n"
       << " C: Move Down\n"
       << " ESC: Exit\n" << endl;

  // --- Main Interaction Loop ---
  while (true) {
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
     // --- Movement ---
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

     // --- Roll ---
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
   }

   // --- Render and Display ---
   Mat frame = cameraEngine.renderFrame();

   // Stop timer and calculate FPS
   auto stop = chrono::high_resolution_clock::now();
   auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
   // Avoid division by zero if duration is very small
   double fps = (duration.count() > 0) ? 1000.0 / duration.count() : 2000.0;

   // Get current camera params for display
   CameraEngine::CameraParams& cameraParams = cameraEngine.getCameraParams();
   
   // Display FPS and camera info on the frame
   string infoText = "FPS: " + to_string(static_cast<int>(fps));
   putText(frame, infoText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
   string posText = "Pos: (" + to_string(cameraParams.position.x).substr(0,4) + ", "
                    + to_string(cameraParams.position.y).substr(0,4) + ", "
                    + to_string(cameraParams.position.z).substr(0,4) + ")";
   putText(frame, posText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
   string rotText = "Pan:" + to_string(static_cast<int>(cameraParams.pan))
                   + " Tilt:" + to_string(static_cast<int>(cameraParams.tilt))
                   + " Roll:" + to_string(static_cast<int>(cameraParams.roll));
   putText(frame, rotText, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);

   imshow("Rendered Frame - Interactive Control", frame);
  } // End main loop

  // Release resources
  destroyAllWindows();
  cout << "Window closed. Application finished." << endl;

  return 0;
}

int main(int argc, char** argv) {

    // Open video from path:
    std::string video_path = "/home/joao/Downloads/IMG_4108.MOV";
    cv::VideoCapture cap(video_path);
    // Open the default webcam (index 0)
    // cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening webcam" << std::endl; // Updated error message
        return -1;
    }
    
    // Get original video frame rate
    double fps = cap.get(cv::CAP_PROP_FPS);
    double frame_period_ms = 1000.0 / fps; // Frame period in milliseconds
    
    std::cout << "Original video FPS: " << fps << ", frame period: " 
              << frame_period_ms << "ms" << std::endl;
    
    // Create stabilizer
    const int past_frames = 2.0 * fps;
    const int future_frames = 1.0 * fps;
    const int working_height = 540;
    Stabilizer stabilizer(past_frames, future_frames, working_height);
    // Tracker tracker; // Tracker logic removed for now
    
    cv::Mat frame;
    // bool trackerInitialized = false; 
    // StabilizationMode currentMode = StabilizationMode::GLOBAL_SMOOTHING;

    // Read first frame to know image size
    cap.read(frame);
    cv::Size frame_size = frame.size();
    std::cout << "Frame size: " << frame_size << std::endl;
    
    // Buffer to store original frames for delay matching
    std::deque<cv::Mat> originalFrameBuffer;

    const std::string ORIGINAL_WINDOW_TITLE = "Original";
    const std::string STABILIZED_WINDOW_TITLE = "Stabilised";
    cv::namedWindow(ORIGINAL_WINDOW_TITLE, cv::WINDOW_KEEPRATIO);
    cv::namedWindow(STABILIZED_WINDOW_TITLE, cv::WINDOW_KEEPRATIO);
    
    // Display help information
    std::cout << "Keyboard controls:" << std::endl;
    std::cout << "  q - Quit" << std::endl;
    std::cout << "  x - Reset stabilizer" << std::endl;
    std::cout << "  f - Full lock mode (freeze frame)" << std::endl;
    std::cout << "  t - Translation lock mode" << std::endl;
    std::cout << "  r - Rotation lock mode" << std::endl;
    std::cout << "  g - Global smoothing mode (allow slow movements)" << std::endl;
    
    bool should_quit = false;

    while (cap.read(frame) && !should_quit) {
        // Start timing this frame's processing
        auto start_time = stabilizer.now();
        
        cv::Mat stabilized;
        
        // --- ROI Selection Logic (Removed) --- 

        // --- Stabilization --- 
        stabilized = stabilizer.stabilizeFrame(frame); // Call the simplified stabilizeFrame
        
        // Store original frame in buffer
        originalFrameBuffer.push_back(frame.clone());
        
        // Display side-by-side only when we have enough frames to sync
        if (originalFrameBuffer.size() > future_frames) {
            // Get the delayed frame that matches the current stabilized frame
            cv::Mat delayedOriginal = originalFrameBuffer.front();
            originalFrameBuffer.pop_front();
            
            // Create side-by-side display
            cv::Mat sideBySide;
            cv::hconcat(delayedOriginal, stabilized, sideBySide);
            
            // --- Display --- 
            cv::imshow(ORIGINAL_WINDOW_TITLE, delayedOriginal);
            cv::imshow(STABILIZED_WINDOW_TITLE, stabilized);
            
            // Calculate elapsed processing time
            auto end_time = stabilizer.now();
            auto processing_time_ms = stabilizer.toMilliseconds(start_time, end_time);
            
            // Calculate adjusted wait time
            int wait_time = static_cast<int>(
                std::max(1.0, frame_period_ms - processing_time_ms.count()));
            
            // --- Key Presses (with dynamic timing) --- 
            char key = cv::waitKey(wait_time);
            switch (key) {
                case 'q':
                    should_quit = true;
                    break;
                case 'x': // Reset stabilizer
                    stabilizer.reset(); 
                    // Clear frame buffer too
                    originalFrameBuffer.clear();
                    // tracker.reset();
                    // trackerInitialized = false;
                    break;
                case 'f':
                    stabilizer.setStabilizationMode(StabilizationMode::FULL_LOCK);
                    break;
                case 't':
                    stabilizer.setStabilizationMode(StabilizationMode::TRANSLATION_LOCK);
                    break;
                case 'r':
                    stabilizer.setStabilizationMode(StabilizationMode::ROTATION_LOCK);
                    break;
                case 'g':
                    stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
                    break;
            }
        } else {
            // We're still filling the buffer, show processing status
            std::cout << "Buffering frames: " << originalFrameBuffer.size() 
                      << "/" << future_frames + 1 << "\r" << std::flush;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
