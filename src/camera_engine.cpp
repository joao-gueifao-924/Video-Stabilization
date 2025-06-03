#include "../include/camera_engine.hpp"

using namespace cv;
using namespace std;

// Constructor
CameraEngine::CameraEngine(const std::string& floorTexturePath) 
    : m_moveSpeed(0.1), m_rollSpeed(2.0) {
    // Load floor texture
    m_floorTexture = imread(floorTexturePath);
    if (m_floorTexture.empty()) {
        cerr << "Error: Could not load floor texture image from '" << floorTexturePath << "'." << endl;
        cerr << "Please ensure the image file exists and is accessible." << endl;
        throw runtime_error("Failed to load floor texture");
    }
    cout << "Floor texture loaded successfully (" 
         << m_floorTexture.cols << "x" << m_floorTexture.rows << ")" << endl;

    // Initialize camera with defaults
    CameraParams defaultParams;
    initCamera(defaultParams);
}

// Destructor
CameraEngine::~CameraEngine() {
    // No specific cleanup needed
}

// Initialize camera parameters
void CameraEngine::initCamera(const CameraParams& params) {
    m_cameraParams = params;
}

// Helper method: Create rotation matrix from pan, tilt, and roll angles
Mat CameraEngine::rotationMatrix(double pan, double tilt, double roll) {
    // Convert angles to radians
    double panRad = degreesToRadians(pan); // Yaw around Y
    double tiltRad = degreesToRadians(tilt); // Pitch around X
    double rollRad = degreesToRadians(roll); // Roll around Z

    // Rotation matrices for each axis
    Mat rotationY = (Mat_<double>(3, 3) << // Pan (Yaw)
        cos(panRad), 0, sin(panRad),
        0, 1, 0,
        -sin(panRad), 0, cos(panRad));

    Mat rotationX = (Mat_<double>(3, 3) << // Tilt (Pitch)
        1, 0, 0,
        0, cos(tiltRad), -sin(tiltRad),
        0, sin(tiltRad), cos(tiltRad));

    Mat rotationZ = (Mat_<double>(3, 3) << // Roll
        cos(rollRad), -sin(rollRad), 0,
        sin(rollRad), cos(rollRad), 0,
        0, 0, 1);

    // Combine rotations: R = Rz(roll) * Rx(tilt) * Ry(pan)
    // This matrix transforms points from camera coordinates to world coordinates
    return rotationZ * rotationX * rotationY;
}

// Constructor for RenderPixelLoopBody
CameraEngine::RenderPixelLoopBody::RenderPixelLoopBody(
    Mat& _frame, const Mat& _floorTexture,
    const Mat& _cameraRotation, const Point3d& _cameraPosition,
    double _focalLength, double _cx, double _cy)
    : frame(_frame), floorTexture(_floorTexture), cameraRotation(_cameraRotation),
    cameraPosition(_cameraPosition), focalLength(_focalLength), cx(_cx), cy(_cy),
    textureCols(_floorTexture.cols), textureRows(_floorTexture.rows) {}

// Parallel loop body implementation
void CameraEngine::RenderPixelLoopBody::operator()(const Range& range) const {
    // Pre-calculate components of camera position
    double camPosX = cameraPosition.x;
    double camPosY = cameraPosition.y;
    double camPosZ = cameraPosition.z;
    
    // Light blue sky color
    const Vec3b skyColor(230, 216, 173); // BGR for light blue

    // Calculate texture aspect ratio
    double textureAspectRatio = static_cast<double>(textureCols) / static_cast<double>(textureRows);
    
    // Calculate tile size in world units
    // Each tile in the world is 1.0 units wide and (1.0/aspectRatio) units high
    double tileWidth = 1.0;
    double tileHeight = tileWidth / textureAspectRatio;

    // Get pointers to rotation matrix elements for slightly faster access
    const double* R = cameraRotation.ptr<double>(0);
    // R = [R00 R01 R02]
    //     [R10 R11 R12]
    //     [R20 R21 R22]

    for (int y = range.start; y < range.end; ++y) {
        Vec3b* frameRowPtr = frame.ptr<Vec3b>(y);
        for (int x = 0; x < frame.cols; ++x) {
            // Pixel coordinates in the image plane relative to the center
            double u = x - cx;
            double v = y - cy;

            // 3D direction vector in camera coordinates (normalized)
            double magnitude = sqrt(u * u + v * v + focalLength * focalLength);
            double cam_dx = u / magnitude;
            double cam_dy = v / magnitude;
            double cam_dz = focalLength / magnitude;

            // Convert direction vector to world coordinates using the pre-calculated rotation matrix
            // worldDir = cameraRotation * camDir
            double dx = R[0] * cam_dx + R[1] * cam_dy + R[2] * cam_dz; // R00*u + R01*v + R02*f
            double dy = R[3] * cam_dx + R[4] * cam_dy + R[5] * cam_dz; // R10*u + R11*v + R12*f
            double dz = R[6] * cam_dx + R[7] * cam_dy + R[8] * cam_dz; // R20*u + R21*v + R22*f

            // Calculate intersection with the floor plane (z = 0)
            // Check if the ray is parallel to the plane or points away from it (dz has same sign as camPosZ)
            // We need t = -camPosZ / dz to be positive (intersection in front of camera ray origin)
            if (abs(dz) < 1e-9 || dz * camPosZ >= 0) {
                frameRowPtr[x] = skyColor; // Set sky color
                continue;
            }

            double t = -camPosZ / dz;

            // 3D point in world coordinates on the floor plane
            double worldX = camPosX + t * dx;
            double worldY = camPosY + t * dy;

            // Convert world coordinates to tile coordinates
            // Using floor and mod to avoid mirroring/reflection artifacts
            double tileX = worldX / tileWidth;
            double tileY = worldY / tileHeight;
            
            // Get the fractional part properly, maintaining sign
            // fmod() can return negative values, we add 1.0 and take fmod again to ensure positive [0,1) range
            double texU = fmod(fmod(tileX, 1.0) + 1.0, 1.0);
            double texV = fmod(fmod(tileY, 1.0) + 1.0, 1.0);
            
            // Invert V coordinate to fix upside-down orientation
            //texV = 1.0 - texV;
            
            // Map to integer texture coordinates
            int textureX = static_cast<int>(texU * textureCols);
            int textureY = static_cast<int>(texV * textureRows);

            // Clamp texture coordinates to be safe
            textureX = max(0, min(textureX, textureCols - 1));
            textureY = max(0, min(textureY, textureRows - 1));

            // Get the color from the floor texture
            frameRowPtr[x] = floorTexture.at<Vec3b>(textureY, textureX);
        }
    }
}

// Render a frame with the current camera parameters
Mat CameraEngine::renderFrame() {
    Mat frame = Mat::zeros(m_cameraParams.sensorResolution, CV_8UC3); // Initialize black
    double focalLength = m_cameraParams.focalLength;
    double cx = m_cameraParams.sensorResolution.width / 2.0;
    double cy = m_cameraParams.sensorResolution.height / 2.0;
    Mat cameraRotation = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d cameraPosition = m_cameraParams.position;

    RenderPixelLoopBody loopBody(frame, m_floorTexture, cameraRotation, cameraPosition,
                                focalLength, cx, cy);
    parallel_for_(Range(0, frame.rows), loopBody);

    return frame;
}

// Camera movement methods
void CameraEngine::moveForward(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldZAxis(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2)); // Camera Forward direction
    m_cameraParams.position += worldZAxis * (amount * m_moveSpeed);
}

void CameraEngine::moveBackward(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldZAxis(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2)); // Camera Forward direction
    m_cameraParams.position -= worldZAxis * (amount * m_moveSpeed);
}

void CameraEngine::moveLeft(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldXAxis(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0)); // Camera Right
    m_cameraParams.position -= worldXAxis * (amount * m_moveSpeed);
}

void CameraEngine::moveRight(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldXAxis(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0)); // Camera Right
    m_cameraParams.position += worldXAxis * (amount * m_moveSpeed);
}

void CameraEngine::moveUp(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldYAxis(R.at<double>(0, 1), R.at<double>(1, 1), R.at<double>(2, 1)); // Camera Up
    worldYAxis = -worldYAxis; // the Y axis of the camera is pointing down
    m_cameraParams.position += worldYAxis * (amount * m_moveSpeed);
}

void CameraEngine::moveDown(double amount) {
    Mat R = rotationMatrix(m_cameraParams.pan, m_cameraParams.tilt, m_cameraParams.roll);
    Point3d worldYAxis(R.at<double>(0, 1), R.at<double>(1, 1), R.at<double>(2, 1)); // Camera Up
    worldYAxis = -worldYAxis; // the Y axis of the camera is pointing down
    m_cameraParams.position -= worldYAxis * (amount * m_moveSpeed);
}

void CameraEngine::rollClockwise(double amount) {
    m_cameraParams.roll -= amount * m_rollSpeed; // Positive roll angle is CCW (standard for Z-axis rotation matrix), so decrease for CW
}

void CameraEngine::rollCounterClockwise(double amount) {
    m_cameraParams.roll += amount * m_rollSpeed; // Positive roll angle is CCW, so increase for CCW
} 