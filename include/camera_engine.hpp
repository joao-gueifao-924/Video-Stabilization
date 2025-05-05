#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

class CameraEngine {
public:
    // Structure to hold camera parameters
    struct CameraParams {
        cv::Point3d position = cv::Point3d(0, 0, 1.0); // Default 1m above the floor
        double pan = 0.0;    // degrees
        double tilt = 105.0;   // degrees
        double roll = 180.0;   // degrees
        double focalLength = 1000.0; // pixels
        cv::Size sensorResolution = cv::Size(1280, 720); // Default HD resolution
    };

    // Constructor
    CameraEngine(const std::string& floorTexturePath, double floorWidth = 10.0);
    ~CameraEngine();

    // Initialize the camera with default parameters
    void initCamera(const CameraParams& params);

    // Render a frame with the current camera parameters
    cv::Mat renderFrame();

    // Camera control methods
    void moveForward(double amount);
    void moveBackward(double amount);
    void moveLeft(double amount);
    void moveRight(double amount);
    void moveUp(double amount);
    void moveDown(double amount);
    void rollClockwise(double amount);
    void rollCounterClockwise(double amount);

    // Getters and setters
    CameraParams& getCameraParams() { return m_cameraParams; }
    void setCameraParams(const CameraParams& params) { m_cameraParams = params; }
    double getMoveSpeed() const { return m_moveSpeed; }
    void setMoveSpeed(double speed) { m_moveSpeed = speed; }
    double getRollSpeed() const { return m_rollSpeed; }
    void setRollSpeed(double speed) { m_rollSpeed = speed; }

private:
    // Helper methods
    static inline double degreesToRadians(double degrees);
    static cv::Mat rotationMatrix(double pan, double tilt, double roll);

    // Class for parallel rendering
    class RenderPixelLoopBody : public cv::ParallelLoopBody {
    private:
        cv::Mat& frame;
        const cv::Mat& floorTexture;
        const cv::Mat cameraRotation;
        const cv::Point3d cameraPosition;
        const double focalLength;
        const double cx, cy;
        const double minFloorX, maxFloorX, minFloorY, maxFloorY;
        const int textureCols, textureRows;

    public:
        RenderPixelLoopBody(cv::Mat& _frame, const cv::Mat& _floorTexture,
            const cv::Mat& _cameraRotation, const cv::Point3d& _cameraPosition,
            double _focalLength, double _cx, double _cy,
            double _minFloorX, double _maxFloorX, double _minFloorY, double _maxFloorY);

        virtual void operator()(const cv::Range& range) const override;
    };

    // Member variables
    CameraParams m_cameraParams;
    cv::Mat m_floorTexture;
    double m_moveSpeed;
    double m_rollSpeed;
    double m_floorWidth;
    double m_minFloorX, m_maxFloorX, m_minFloorY, m_maxFloorY;
}; 