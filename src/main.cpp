#include "stabilizer.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

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
    
    // Create stabilizer
    const int window_size = 30;
    Stabilizer stabilizer(window_size);
    // Tracker tracker; // Tracker logic removed for now
    
    cv::Mat frame;
    // bool trackerInitialized = false; 
    // StabilizationMode currentMode = StabilizationMode::GLOBAL_SMOOTHING;
    
    while (cap.read(frame)) {
        cv::Mat stabilized;
        
        // --- ROI Selection Logic (Removed) --- 

        // --- Stabilization --- 
        stabilized = stabilizer.stabilizeFrame(frame); // Call the simplified stabilizeFrame
        
        // --- Display --- 
        cv::imshow("Stabilized Video", stabilized);
        
        // --- Key Presses (Simplified) --- 
        const int delay_ms = 20;
        char key = cv::waitKey(delay_ms);
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
