#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief Manages a virtual 3D camera and renders scenes with a textured floor.
 *
 * The CameraEngine class encapsulates the logic for a virtual camera in a 3D 
 * environment. It handles camera parameters such as position, orientation 
 * (pan, tilt, roll), focal length, and sensor resolution. The primary function 
 * of this class is to render frames of a scene, which consists of an infinitely 
 * tiled floor (based on a provided texture) and a simple sky color.
 *
 * Key functionalities include:
 * - Initialization of camera parameters with default or custom values.
 * - Loading and managing a floor texture. The texture is mapped to a canonical
 *   1.0 world unit wide tile, with its height determined by the texture's aspect ratio
 *   to prevent distortion. This tile is then repeated infinitely.
 * - Rendering frames through a ray tracing approach for each pixel.
 * - Providing methods to control the camera's movement (forward, backward, left, 
 *   right, up, down) and orientation (roll).
 * - Management of movement and roll speeds.
 *
 * The rendering process calculates the color for each pixel by determining if the
 * corresponding ray intersects with the floor or points towards the sky. The class
 * utilizes OpenCV for image manipulation and matrix operations. It also includes
 * a parallelized loop for faster frame rendering.

 * @note The rendering process is CPU-based and computationally intensive, 
 * as it does not utilize GPU acceleration.
 * @note This class serves as a debugging tool for the VideoStabilizer class by 
 * providing precise, repeatable, and fully controllable camera movements and 
 * orientations.
 */
class CameraEngine {
public:
    /**
     * @brief Structure to hold camera parameters.
     *
     * This structure encapsulates all parameters that define the camera's state,
     * including its position in 3D space, orientation (pan, tilt, roll),
     * focal length, and sensor resolution.
     */
    struct CameraParams {
        cv::Point3d position = cv::Point3d(0, 0, 1.0); ///< Camera position in world coordinates (meters). Default is 1 meter above the origin.
        double pan = 0.0;    ///< Pan angle in degrees (yaw, rotation around the Y-axis).
        double tilt = 105.0;   ///< Tilt angle in degrees (pitch, rotation around the X-axis).
        double roll = 180.0;   ///< Roll angle in degrees (rotation around the Z-axis, the optical centre). Positive angle results in counter-clockwise rotation.
        double focalLength = 1000.0; ///< Focal length of the camera in pixels.
        cv::Size sensorResolution = cv::Size(1280, 720); ///< Resolution of the camera sensor (width, height) in pixels. Default is HD (1280x720).

        /**
         * @brief Default constructor.
         * Initializes camera parameters with default values.
         */
        CameraParams() = default;

        /**
         * @brief Parameterized constructor.
         * Initializes camera parameters with specified values.
         * @param position Camera position in (X,Y,Z) world coordinates (meters).
         * @param pan Pan angle in degrees.
         * @param tilt Tilt angle in degrees.
         * @param roll Roll angle in degrees.
         * @param focalLength Focal length in pixels.
         * @param sensorResolution Sensor resolution in pixels (width, height).
         */
        CameraParams(cv::Point3d position, double pan, double tilt, double roll,
                    double focalLength, cv::Size sensorResolution)
            : position(position), pan(pan), tilt(tilt), roll(roll),
              focalLength(focalLength), sensorResolution(sensorResolution) {}
    };

    // Constructor
    /**
     * @brief Constructs a CameraEngine object.
     * Initializes the camera engine, loads the floor texture, and sets up initial parameters.
     * The floor texture image is mapped to a canonical 1.0 world unit wide tile. This tile's
     * height is determined by the texture's aspect ratio to prevent distortion (height = 1.0 / aspect_ratio).
     * This canonical tile is then repeated infinitely to form the floor.
     * @param floorTexturePath Path to the image file for the floor texture.
     * @throws std::runtime_error if the floor texture cannot be loaded.
     */
    CameraEngine(const std::string& floorTexturePath);
    ~CameraEngine();

    // Initialize the camera with default parameters
    /**
     * @brief Initializes the camera with a given set of parameters.
     * @param params A CameraParams struct containing the desired camera settings.
     */
    void initCamera(const CameraParams& params);

    // Render a frame with the current camera parameters
    /**
     * @brief Renders a single frame based on the current camera parameters and floor texture.
     * This method performs ray tracing for each pixel to determine its color,
     * rendering either the floor texture or a sky color.
     * @return cv::Mat The rendered frame as an OpenCV matrix.
     */
    cv::Mat renderFrame();

    // Camera control methods
    /**
     * @brief Moves the camera forward along its viewing direction.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveForward(double amount);

    /**
     * @brief Moves the camera backward, opposite to its viewing direction.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveBackward(double amount);

    /**
     * @brief Moves the camera left, perpendicular to its viewing direction and up vector.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveLeft(double amount);

    /**
     * @brief Moves the camera right, perpendicular to its viewing direction and up vector.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveRight(double amount);

    /**
     * @brief Moves the camera upward along its local up vector.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveUp(double amount);

    /**
     * @brief Moves the camera downward along its local up vector.
     * @param amount The amount to move, scaled by the current move speed.
     */
    void moveDown(double amount);

    /**
     * @brief Rolls the camera clockwise around its viewing (Z) axis.
     * @param amount The amount to roll in degrees, scaled by the current roll speed.
     */
    void rollClockwise(double amount);

    /**
     * @brief Rolls the camera counter-clockwise around its viewing (Z) axis.
     * @param amount The amount to roll in degrees, scaled by the current roll speed.
     */
    void rollCounterClockwise(double amount);

    // Getters and setters
    /**
     * @brief Gets a reference to the current camera parameters.
     * @return CameraParams& A reference to the internal CameraParams struct.
     */
    CameraParams& getCameraParams() { return m_cameraParams; }

    /**
     * @brief Sets the camera parameters.
     * @param params A CameraParams struct containing the new camera settings.
     */
    void setCameraParams(const CameraParams& params) { m_cameraParams = params; }

    /**
     * @brief Gets the current camera movement speed.
     * @return double The movement speed that scales all camera movement.
     *         This speed multiplier affects moveForward(), moveRight(), moveUp(), etc.
     *         A higher speed means the camera will move further for a given
     *         movement amount.
     */
    double getMoveSpeed() const { return m_moveSpeed; }

    /**
     * @brief Sets the camera movement speed.
     * @param speed The new movement speed multiplier that will scale all
     *              camera movement amounts. This affects how far the camera
     *              moves when calling any of the movement methods like
     *              moveForward(), moveLeft(), moveUp(), etc. A higher speed
     *              means more movement per method call.
     */
    void setMoveSpeed(double speed) { m_moveSpeed = speed; }

    /**
     * @brief Gets the current camera roll speed.
     * @return double The roll speed multiplier that scales all camera roll rotations.
     *         This speed multiplier affects rollClockwise() and rollCounterClockwise().
     *         A higher speed means the camera will rotate further for a given
     *         roll amount.
     */
    double getRollSpeed() const { return m_rollSpeed; }

    /**
     * @brief Sets the camera roll speed.
     * @param speed The new roll speed multiplier that will scale all
     *              camera roll rotations. This affects how far the camera
     *              rotates when calling rollClockwise() or rollCounterClockwise().
     *              A higher speed means more rotation per method call.
     */
    void setRollSpeed(double speed) { m_rollSpeed = speed; }

private:
    // Helper methods
    static inline double degreesToRadians(double degrees) {
        return degrees * M_PI / 180.0;
    }
    /**
     * @brief Calculates a 3D rotation matrix from pan, tilt, and roll angles.
     * @param pan The pan angle in degrees (yaw, rotation around Y-axis)
     * @param tilt The tilt angle in degrees (pitch, rotation around X-axis) 
     * @param roll The roll angle in degrees (rotation around Z-axis)
     * @return cv::Mat A 3x3 rotation matrix combining the pan, tilt and roll rotations
     * 
     * @note The rotations are applied in the order: pan, then tilt, then roll.
     *       All angles should be provided in degrees and will be converted to radians internally.
     */
    static cv::Mat rotationMatrix(double pan, double tilt, double roll);

    // Class for parallel rendering
    /**
     * @brief Implements parallel processing for rendering pixels.
     *
     * This class extends `cv::ParallelLoopBody` to enable parallel rendering of image pixels.
     * It processes a range of pixels, calculating the color for each based on ray tracing
     * and the intersection with the floor plane or sky. The floor texture is rendered
     * with an effectively infinite tiling pattern.
     */
    class RenderPixelLoopBody : public cv::ParallelLoopBody {
    private:
        cv::Mat& frame; ///< Reference to the output frame being rendered.
        const cv::Mat& floorTexture; ///< Reference to the floor texture image.
        const cv::Mat cameraRotation; ///< Camera rotation matrix.
        const cv::Point3d cameraPosition; ///< Camera position in world coordinates.
        const double focalLength; ///< Focal length of the camera.
        const double cx, cy; ///< Principal point coordinates (center of the sensor).
        const int textureCols, textureRows; ///< Dimensions of the floor texture.

    public:
        /**
         * @brief Constructs a RenderPixelLoopBody object.
         * @param _frame Reference to the output frame.
         * @param _floorTexture Reference to the floor texture.
         * @param _cameraRotation Camera rotation matrix.
         * @param _cameraPosition Camera position.
         * @param _focalLength Camera focal length.
         * @param _cx X-coordinate of the principal point.
         * @param _cy Y-coordinate of the principal point.
         */
        RenderPixelLoopBody(cv::Mat& _frame, const cv::Mat& _floorTexture,
            const cv::Mat& _cameraRotation, const cv::Point3d& _cameraPosition,
            double _focalLength, double _cx, double _cy);

        virtual void operator()(const cv::Range& range) const override;
    };

    // Member variables
    CameraParams m_cameraParams;
    cv::Mat m_floorTexture;
    double m_moveSpeed;
    double m_rollSpeed;
};
