#include "stabilizer.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // For timing measurements
#include <deque> // For frame buffer

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
    std::cout << "  r - Reset stabilizer" << std::endl;
    std::cout << "  f - Full lock mode (freeze frame)" << std::endl;
    std::cout << "  g - Global smoothing mode (allow slow movements)" << std::endl;
    
    while (cap.read(frame)) {
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
            if (key == 'q') {
                break;
            } else if (key == 'r') { // Reset stabilizer 
                stabilizer.reset(); 
                // Clear frame buffer too
                originalFrameBuffer.clear();
                // tracker.reset();
                // trackerInitialized = false;
            } else if (key == 'f') {
                stabilizer.setStabilizationMode(StabilizationMode::FULL_LOCK);
            } else if (key == 'g') {
                stabilizer.setStabilizationMode(StabilizationMode::GLOBAL_SMOOTHING);
            }
            // Removed other mode-switching keys (d, f, t, o, g) for now...
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
