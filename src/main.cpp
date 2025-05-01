#include "stabilizer.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono> // For timing measurements

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
    int frame_period_ms = 1000.0 / fps; // Frame period in milliseconds
    std::cout << "Original video FPS: " << fps << ", frame period: " << frame_period_ms << "ms" << std::endl;
    
    // Create stabilizer
    const int window_size = 30;
    const int working_height = 540;
    Stabilizer stabilizer(window_size, working_height);
    // Tracker tracker; // Tracker logic removed for now
    
    cv::Mat frame;
    // bool trackerInitialized = false; 
    // StabilizationMode currentMode = StabilizationMode::GLOBAL_SMOOTHING;

    // Read first frame to know image size
    cap.read(frame);
    cv::Size frame_size = frame.size();
    std::cout << "Frame size: " << frame_size << std::endl;
    
    while (cap.read(frame)) {
        // Start timing this frame's processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cv::Mat stabilized;
        
        // --- ROI Selection Logic (Removed) --- 

        // --- Stabilization --- 
        stabilized = stabilizer.stabilizeFrame(frame); // Call the simplified stabilizeFrame
        
        // --- Display --- 
        cv::imshow("Stabilized Video", stabilized);
        
        // Calculate elapsed processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        // Calculate adjusted wait time
        int wait_time = static_cast<int>(std::max(1.0, frame_period_ms - static_cast<double>(processing_time_us) / 1000.0));
        
        // --- Key Presses (with dynamic timing) --- 
        char key = cv::waitKey(wait_time);
        if (key == 'q') {
            break;
        } else if (key == 'r') { // Reset stabilizer 
            stabilizer.reset(); 
            // tracker.reset();
            // trackerInitialized = false;
        }
        // Removed other mode-switching keys (d, f, t, o, g) for now...
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
