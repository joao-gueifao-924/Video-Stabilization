# Real-time Video Stabilization with Selective Motion Suppression via Homography Decomposition

A real-time video stabilization system implementing multiple computer vision algorithms for camera motion estimation and suppression. Camera motion can be handled in several ways: reducing overall shake while preserving intentional movement, completely locking the view in place, or selectively canceling only camera translation or rotation. The program has support for live video streams, pre-recorded footage, and a 3D camera simulator for development and testing. Designed for real-time performance without GPU acceleration, the implementation uses traditional computer vision approaches rather than deep learning methods.

While this README is focused on practical usage instructions, a detailed explanation of the underlying video stabilization algorithms can be found in the [Mathematical Overview](https://joao-gueifao-924.github.io/Video-Stabilization/math-overview.html) page. Refer to this document for a comprehensive theoretical background on the methods implemented in this project.

This implementation prioritizes real-time performance and educational value, demonstrating multiple approaches to video stabilization in a single, comprehensive system.

## Features
This project provides a flexible, real-time video stabilization toolkit with several capabilities, listed below:


### Stabilization Modes
This project offers several stabilization modes to suit different needs.

- **Global Smoothing**: Reduces camera shake by averaging camera motion over a sliding temporal window while preserving intentional camera movement
- **Full Motion Lock**: Completely cancels out camera motion using three different approaches, freezing view of the scenery:
  - **Accumulated Optical Flow**: Uses sparse Lucas-Kanade optical flow with frame-to-frame transformation accumulation
  - **ORB-based Registration**: Employs Oriented FAST and Rotated BRIEF features for direct frame alignment between presented frames and a reference frame
  - **SIFT-based Registration**: Same as for *ORB-based Registration* but uses Scale-Invariant Feature Transform instead for highest accuracy in frame registration
- **Translation Lock**: Compensates only horizontal and vertical camera movement (preserves rotation) *[Not fully implemented yet]*
- **Rotation Lock**: Compensates only camera rotation around the image center (preserves translation) *[Not fully implemented yet]*

### Input Sources
The system supports multiple input sources for flexible experimentation and different use cases:

- **Live Camera**: Real-time webcam input for immediate stabilization
- **Video Files**: Processing of pre-recorded video files (MP4, AVI, MOV, etc.)
- **3D Simulator**: Synthetic scene and camera simulation with controllable movement over textured floor scenes

### Additional Features
- Real-time performance
- Configurable temporal windows for motion analysis
- Side-by-side comparison of original and stabilized video
- Interactive runtime controls for switching between stabilization modes
- Timing statistics and performance monitoring
- Image preprocessing for enhanced detection of visual features
- Optional ECC (Enhanced Correlation Coefficient) refinement capability *(currently disabled)*

## Requirements

- **OpenCV 4.x** - Computer vision library
- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)  
- **CMake 3.10** or higher
- **Linux/macOS/Windows** (cross-platform compatible)

## Building

```bash
# Clone the repository
git clone <repository-url>
cd Video-Stabilization

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make

# Or for faster parallel build
make -j$(nproc)
```

### Build Options
```bash
# Debug build with optimization disabled
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
```


## Usage

### Command Line Syntax
```bash
./video_stabilization [INPUT_MODE] [OPTIONS]

# Show help
./video_stabilization --help
./video_stabilization -h
```

### Input Modes (exactly one required)

#### Camera Input
```bash
./video_stabilization --camera [CAMERA_ID]

# Examples:
./video_stabilization --camera 0          # Default webcam
./video_stabilization --camera 1          # Secondary camera
```

#### Video File Input  
```bash
./video_stabilization --file [VIDEO_PATH]

# Examples:
./video_stabilization --file video.mp4
./video_stabilization --file /path/to/footage.avi
```

#### Simulator Input
```bash
./video_stabilization --simulator [TEXTURE_IMAGE_PATH]

# Examples:
./video_stabilization --simulator floor_texture.jpg
./video_stabilization --simulator textures/brick_pattern.png

# Generates synthetic camera movement over the provided texture
# Useful for testing and algorithm development
```

### Optional Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--past-window SECONDS` | Amount of video time prior to the presented frame, in seconds | 2.0 | ≥ 0.0 |
| `--future-window SECONDS` | Amount of video time ahead of the presented frame, in seconds | 1.5 | ≥ 0.0 |
| `--working-height PIXELS` | Processing resolution height | 360 | > 90 and ≤ 2160 |

**Important**: Total window size (`--past-window` + `--future-window`) must be ≥ 0.030 seconds.
Each of the optional parameters allows you to tune the stabilization process for your specific needs. Here’s what they do and the trade-offs involved:

- **`--past-window SECONDS`**  
  *Effect*: Sets how much video time before the current frame is used for stabilization.  
  *Trade-off*: Increasing this value allows the algorithm to use more historical context, which can improve stability and smoothness. However, using a large past window can result in excessive smoothing, especially for very jittery footage. This may cause a significant portion of the stabilized video frame to fall outside the visible canvas, leading to more aggressive cropping or black borders. There is **no added presentation latency** from increasing this value.

- **`--future-window SECONDS`**  
  *Effect*: Sets how much video time after the current frame is used for stabilization.  
  *Trade-off*: A larger future window allows the algorithm to "look ahead" and better anticipate motion, resulting not only in smoother output but also in reducing the lag between actual camera movement and the stabilization's response to changes in motion speed. However, as with the past window, excessive smoothing lead too aggressive cropping or black borders, especially with very unstable footage. For live input (e.g., webcam), keep this value low to minimize display latency, since the program must wait for future frames before showing the stabilized result.

- **`--working-height PIXELS`**  
  *Effect*: Determines the vertical resolution at which the stabilization algorithm processes frames.  
  *Trade-off*: Higher values improve stabilization accuracy and output quality but require more computation and may slow down processing. Lower values speed up processing and reduce resource usage, but may result in less precise stabilization. Choose a value that balances quality and performance for your hardware and use case.

**Summary:**  
- For **real-time use** (e.g., webcam), use a small `--future-window` to keep display delay low, and a moderate `--working-height` for speed.  
- For **offline processing** (e.g., video files), you can increase the window sizes for maximum smoothness, but be aware that excessive smoothing (large windows) may lead to more image cropping or visible borders, especially with very shaky footage. Use a higher working height for best quality if performance allows.


### Complete Examples
```bash
# High-quality stabilization with large temporal window
./video_stabilization --file shaky_video.mp4 --past-window 3.0 --future-window 2.0 --working-height 720

# Real-time webcam with fast processing
./video_stabilization --camera 0 --past-window 1.0 --future-window 0.5 --working-height 240

# Simulator testing with brick texture
./video_stabilization --simulator brick_texture.jpg --working-height 480
```

## Runtime Controls

At startup, the program displays these controls and opens two windows: **"Original Camera Feed"** and **"Stabilized Output"**.
If a future stabilization window (`--future-window`) is used, both feeds remain synchronized—the original feed will be delayed as needed to match the stabilized output.

### Universal Controls (All Modes)
| Key | Action |
|-----|--------|
| `ESC` | Exit program gracefully |

### Camera Movement (Simulator Mode Only)
| Key | Action |
|-----|--------|
| `W` | Move camera forward |
| `S` | Move camera backward |
| `A` | Move camera left (strafe) |
| `D` | Move camera right (strafe) |
| `Q` | Roll counter-clockwise |
| `E` | Roll clockwise |
| `SPACE` | Move camera up |
| `C` | Move camera down |
| `P` | Reset camera to default position |

### Stabilization Controls (All Modes)
| Key | Action |
|-----|--------|
| `X` | Reset to Global Smoothing mode |
| `G` | Global Smoothing stabilization |
| `F` | Full Lock (Optical Flow based) |
| `O` | Full Lock (ORB feature based) |
| `L` | Full Lock (SIFT feature based) |
| `T` | Translation Lock mode *(not fully implemented)* |
| `R` | Rotation Lock mode *(not fully implemented)* |


## Troubleshooting

### Common Issues

**Camera not opening:**
- Check camera permissions and availability
- Try different camera IDs (0, 1, 2, etc.)
- Ensure no other applications are using the camera
- OpenCV (one of the core dependencies) can occasionally encounter issues when initializing camera capture sessions. In particular, its camera handling may not be fully reliable for production environments.

**Video file not loading:**
- Verify file path and permissions
- Supported video formats and codecs depend on your OpenCV build

**Simulator texture not loading:**
- Verify file path and permissions
- Supported image formats depend on your OpenCV build

**Poor stabilization quality:**
- Increase working height for better feature detection
- Adjust temporal window sizes based on motion characteristics
- Try different stabilization modes for your specific use case
- Ensure sufficient texture/features in the scene (minimum 10 trackable points required)
- Ensure adequate lighting and good contrast in the scene for reliable feature detection—this is important for all modes, but especially for ORB and SIFT.
- Lower your camera's shutter speed to minimize motion blur, which is especially common in artificial lighting conditions.
- Ensure your camera has low image noise for optimal feature detection

**Performance issues:**
- Reduce working height for faster processing
- Use smaller temporal windows
- Close other resource-intensive applications
- Consider optical flow mode for better performance

### Debug Information
The program provides real-time performance statistics including:
- Feature detection timing (goodFeaturesToTrack, ORB, SIFT)
- Optical flow computation time  
- Homography estimation duration
- Frame warping performance
- Overall processing FPS
- Feature count and match quality metrics


## Known Limitations

- **Translation Lock** and **Rotation Lock** modes are not correctly implemented yet (marked as TODO)
- Achieving real-time performance depends on using appropriate parameter settings
- ECC refinement is implemented but currently disabled for performance reasons
- Camera motion estimation relies on a rigid-body model, which improves accuracy by reducing the number of degrees of freedom, but may not be suitable for all scenarios.


## Further Reading: Mathematical Overview

For a detailed explanation of the mathematical foundations and algorithms behind this video stabilization system—including homography decomposition, camera motion models, and visual feature detection—please refer to the [Mathematical Overview](https://joao-gueifao-924.github.io/Video-Stabilization/math-overview.html). This document provides in-depth technical background and references for the methods implemented in the project.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, bug reports, or feature requests.

