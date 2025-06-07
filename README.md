# Real-time Video Stabilization with Selective Motion Suppression

A real-time video stabilization system implementing multiple computer vision algorithms for camera motion estimation and suppression. Camera motion can be handled in several ways: reducing overall shake while preserving intentional movement, completely locking the view in place, or selectively canceling only camera translation or rotation. The program has support for live video streams, pre-recorded footage, and a 3D camera simulator for development and testing. Designed for real-time performance without GPU acceleration, the implementation uses traditional computer vision approaches rather than deep learning methods.

## Features

### Stabilization Modes
- **Global Smoothing**: Reduces camera shake by averaging camera motion over a sliding temporal window while preserving intentional camera movement
- **Full Motion Lock**: Completely cancels out camera motion using three different approaches, freezing view of the scenery:
  - **Accumulated Optical Flow**: Uses sparse Lucas-Kanade optical flow with frame-to-frame transformation accumulation
  - **ORB-based Registration**: Employs Oriented FAST and Rotated BRIEF features for direct frame alignment between presented frames and a reference frame
  - **SIFT-based Registration**: Same as for *ORB-based Registration* but uses Scale-Invariant Feature Transform instead for highest accuracy in frame registration
- **Translation Lock**: Compensates only horizontal and vertical camera movement (preserves rotation) *[Not fully implemented yet]*
- **Rotation Lock**: Compensates only camera rotation around the image center (preserves translation) *[Not fully implemented yet]*

### Input Sources
- **Live Camera**: Real-time webcam input for immediate stabilization
- **Video Files**: Processing of pre-recorded video files (MP4, AVI, MOV, etc.)
- **3D Simulator**: Synthetic scene and camera simulation with controllable movement over textured floor scenes

### Advanced Features
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

The program displays these controls at startup and creates two windows: **"Original Camera Feed"** and **"Stabilized Output"**.

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


## Mathematical Overview

This section provides a mathematical overview of the core concepts underlying the video stabilization system. It explains how camera motion is modeled and estimated using homography and more specifically rigid-body transformations, and describes the decomposition of these transformations into fundamental geometric parameters. Understanding these principles is essential for grasping how the stabilizer separates and selectively suppresses different types of camera movement.


### Homography Decomposition into fundamental parameters

Below we cite (with small adaptations) from the popular, highly regarded *Multiple View Geometry in Computer Vision* book [1]:
> A projective transformation can be decomposed into a chain of transformations, where each matrix in the chain represents a transformation higher in the hierarchy than the previous one:  
>
> $$
H = H_S H_A H_P =
\begin{bmatrix}
sR & t \\
0^\top & 1
\end{bmatrix}
\begin{bmatrix}
K & 0 \\
0^\top & 1
\end{bmatrix}
\begin{bmatrix}
I & 0 \\
v^\top & \eta
\end{bmatrix}
=
\begin{bmatrix}
A & t \\
v^\top & \eta
\end{bmatrix}
> $$
>with $A$ a non-singular matrix given by $A = sRK+tv^\top$, and $K$ an upper-triangular matrix normalized as $detK = 1$. This decomposition is valid provided $\eta \neq 0$, and is unique if $s$ is chosen positive.
>
> Each of the matrices $H_S$, $H_A$, $H_P$ is the “essence” of a transformation of that type (as indicated by the subscripts $S$, $A$, $P$). [...] $H_P$ (2 dof) moves the line at infinity; $H_A$ (2 dof) affects the affine properties, but does not move the line at infinity; and finally, $H_S$ is a general similarity transformation (4 dof) which does not affect the affine or projective properties.

In practice, normal frame-to-frame registration with real camera motion produces a non-zero value for $\eta$, the bottom-right entry of the homography matrix. If this entry ever becomes zero, it typically signals a degenerate or pathological case in the estimation process, indicating that the motion estimate has failed. The codebase is designed to handle such likely rare scenarios robustly by substituting the produced estimate by the identity matrix.

We further decompose the homography matrix into fundamental parameters, listed below, each corresponding to one of the elementary matrices described above:

- **Isotropic Scaling** ($s$): Isotropic scaling factor
- **Rotation** ($\theta$): In-image-plane rotation angle
- **Anisotropic Scaling** ($k_1, k_2$): Anisotropic scaling ratio where $k_1 \cdot k_2 = 1$
- **Shear** ($\delta$): Shear transformation value
- **Translation** ($t$): 2D translation vector $t=(t_x, t_y)$
- **Perspective** ($v$): Horizon line shift vector $v=(v_x, v_y)$

A homography transformation between two frames can be expressed in two equivalent forms: as a $3\times3$ matrix $H$ or using the parametric decomposition shown above. Both representations capture the same geometric transformation and can be uniquely converted between each other. Importantly, while the matrix form has $9$ entries, it effectively has $8$ degrees of freedom since homographies are defined up to an arbitrary scale factor – matching the $8$ scalar parameters in the decomposed form: 2 for isotropic scaling $s$ and rotation $θ$, 1 for anisotropic scaling ratio $k_1/k_2$, 1 for shear $\delta$, 2 for translation vector $(t_x, t_y)$, and 2 for perspective vector $(v_x, v_y)$.

The homography decomposition, as implemented in [`decomposeHomography()`](stabilizer.cpp) method, proceeds as follows:

1. **Normalization**: The input $3 \times 3$ homography matrix $H$ is normalized so that the bottom-right entry $H_{3,3}$ becomes $1$.

2. **Extraction of Components**:
   - The translation vector $t$ is extracted from the top-right $2 \times 1$ block of the normalized matrix.
   - The perspective vector $v$ is extracted from the bottom-left $1 \times 2$ block.
   - The upper-left $2 \times 2$ block ($A$) is also extracted.

3. **Projective Correction**:
   - The matrix $A$ is corrected for projective effects by subtracting the outer product of $t$ and $v$, yielding $sRK = A - t \cdot v^\top$.
   - This isolates the affine part of the transformation.

4. **Isotropic Scaling Extraction**:
   The determinant of the $(sRK)$ matrix is computed. Then, we make use of the following mathematical properties: 
   - For some scalar $s$ and a $n \times n$ invertible matrix $M$, we have $\det(s M) = s^n \det(M)$ 
   - $R$ is a $2 \times 2$ rotation matrix, hence has positive unitary determinant, $det(R) = 1$
   - $K$ is a $2 \times 2$  upper triangular matrix also with positive unitary determinant, $det(K)=1$

   The isotropic scaling factor $s$ is then given by  $s = \sqrt{\det(sRK)}$.

5. **QR Decomposition**:
   
   - *QR decomposition* is a mathematical technique that factors a matrix into two components: an orthogonal matrix $Q$ and an upper-triangular matrix $R$. (Note: in this paragraph only, $R$ refers to the upper-triangular matrix from the QR decomposition, not the rotation matrix $R$ used elsewhere in our decomposition — the names simply overlap.)

   - In our implementation, we apply QR decomposition (using the Gram-Schmidt process) to the matrix $(RK) = (sRK)/s$. This allows us to separate $(RK)$ into a true rotation matrix $R$ and an upper-triangular matrix $K$. 

   - Because we are working with $2 \times 2$ matrices, the classical Gram-Schmidt process is both simple and numerically stable enough for our needs, so we use it instead of more complex methods.

6. **Rotation angle, anisotropic scaling and shear parameters extraction**
    - We have $R = \begin{bmatrix}
  cos(\theta) & -sin(\theta) \\
  sin(\theta) & cos(\theta)
  \end{bmatrix}$ and $K = \begin{bmatrix}
  k_1 & \delta \\
  0 & k_2
  \end{bmatrix}$ as output from previous step. We easily extract $\theta$ using `atan2` function on the entries of $R$ and remaining parameters $k_1, k_2, \delta$ directly from the entries of $K$.

7. **Translation Correction**:
   - A similarity (aka. Euclidean transformation) has exactly 1 fixed or invariant point, around which isotropic scaling and rotation take place. Normally, this fixed point is the coordinate system origin $0$.
   - The general form for a similarity, for an arbitrary non-zero fixed point, is:
    $$p' = s[R(p-c)+c]+t = $$
    $$= sRp - sRc + sc + t = $$
    $$= sRp + t + sc - sRc = $$
    $$= sRp + [t + s(I-R)c] = $$
    $$= sRp + \tilde{t}$$
   - The translation vector $\tilde{t}$ captures both camera translational movement and the additional shift resulting from scaling and rotating around the point $c$.
   - OpenCV and other common image processing and computer vision libraries assign the top-left corner of an image as the origin of the coordinate system. In the context of camera motion stabilization, it is more natural to define rotations to be made around the camera optical axis, approximated by the image centre.
   - To ensure the decomposition is centered at the image center $c$, we correct the translation $t$ by removing the effect of scaling and rotating around a non-zero fixed point. Thus, we define $t = \tilde{t} - s (I - R) c$.

This decomposition allows the stabilizer to independently manipulate rotation, translation, scaling, shear, and perspective components of the camera motion, enabling fine-grained stabilization modes.  For detailed mathematical steps and their implementation, see the [`decomposeHomography()`](stabilizer.cpp) method and its counterpart [`composeHomography()`](stabilizer.cpp).


### Stabilization Mathematics

#### Global Smoothing
For a frame at time t with temporal window size W:
```
H_smooth(t) = (1/W) · Σ[i=-W/2 to W/2] H(t,t+i)
```
Where H(t,t+i) is the transformation from frame t to frame t+i.

#### Motion Locking
For full motion cancellation relative to reference frame R:
```
H_stabilize(t) = H(R,t)^(-1)
```
Where H(R,t) transforms from reference frame R to current frame t.

#### Motion Model Constraints
- Uses **2D Rigid-Body** motion model (rotation + translation, no scaling)
- Automatically removes isotropic scaling from estimates to prevent instability
- Center of rotation fixed at image center to avoid scaling artifacts
- RANSAC outlier rejection for robust parameter estimation
- Optional **ECC refinement** available but currently disabled for performance

## Algorithm Performance Comparison

### Optical Flow vs Feature Matching
- **Optical Flow**: Fastest, good for small motions, can drift over time, requires ≥10 tracked points, ~1300 features max
- **ORB Features**: Balanced speed/accuracy, robust to moderate scene changes, up to 2500 features, Hamming distance matching
- **SIFT Features**: Highest accuracy, best for large motions, computationally intensive, up to 2500 features, Flann-based matching

### Temporal Window Effects
- **Larger Past Window**: Smoother stabilization, better noise reduction
- **Larger Future Window**: Reduced lag, improved motion prediction, increased delay
- **Higher Working Resolution**: Better accuracy, increased computation time, more features detected

### Image Preprocessing Pipeline
For ORB and SIFT modes, sophisticated preprocessing enhances feature detection:
1. **Median blur** (5×5 kernel) - noise reduction
2. **Sharpening filter** (3×3 kernel) - edge enhancement  
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - contrast enhancement with 2.0 clip limit, 8×8 tiles
4. **Median blur** (5×5 kernel) - final smoothing

## Implementation Details

### Frame Processing Pipeline
1. **Input Capture**: Grab frame from camera/file/simulator
2. **Preprocessing**: Apply blur, sharpening, and contrast enhancement (ORB/SIFT modes only)
3. **Feature Detection**: Extract corner features or keypoints (filtered by size)
4. **Motion Estimation**: Track features or match descriptors with outlier rejection
5. **Transformation Computation**: Calculate inter-frame homography with scaling removal
6. **Stabilization**: Apply motion compensation based on selected mode
7. **Output Rendering**: Warp frame using computed transformation
8. **Display**: Present original and stabilized frames side-by-side

### Memory Management
- Sliding window buffers for frames and transformations
- Automatic memory cleanup with configurable window sizes
- Efficient OpenCV matrix operations with minimal copying

### Multi-threading
- Parallel pixel processing for 3D simulator rendering
- OpenCV's optimized parallel implementations for computer vision operations
- Non-blocking keyboard input handling

### Technical Parameters
- **QR Decomposition**: Custom 2×2 implementation for homography decomposition
- **RANSAC**: 5.0 pixel reprojection threshold for motion estimation
- **Feature Scaling**: Resolution-adaptive minimum distances and quality levels
- **Border Handling**: 10-pixel border exclusion for warp mask generation

## Troubleshooting

### Common Issues

**Camera not opening:**
- Check camera permissions and availability
- Try different camera IDs (0, 1, 2, etc.)
- Ensure no other applications are using the camera

**Video file not loading:**
- Verify file path and permissions
- Check OpenCV codec support for the video format
- Try converting to MP4 with H.264 encoding

**Simulator texture not loading:**
- Verify the texture image file exists and is readable
- Supported formats: JPG, PNG, BMP, TIFF
- Ensure the image path is correct

**Poor stabilization quality:**
- Increase working height for better feature detection
- Adjust temporal window sizes based on motion characteristics
- Try different stabilization modes for your specific use case
- Ensure sufficient texture/features in the scene (minimum 10 trackable points required)
- For ORB/SIFT modes, ensure adequate lighting and contrast

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

## Technical References

- **Lucas-Kanade Optical Flow**: Shi-Tomasi corner detection with pyramidal LK tracking
- **ORB Features**: Oriented FAST keypoints with rotated BRIEF descriptors  
- **SIFT Features**: Lowe's Scale-Invariant Feature Transform algorithm
- **Homography Estimation**: RANSAC-based robust parameter estimation
- **Motion Models**: Euclidean (rigid-body) and affine transformation fitting
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhanced feature detection
- **QR Decomposition**: Gram-Schmidt process for 2×2 matrix factorization

## Known Limitations

- **Translation Lock** and **Rotation Lock** modes are not correctly implemented yet (marked as TODO)
- Feature-based methods require sufficient scene texture (minimum 10 trackable features)
- Real-time performance depends on scene complexity and parameter settings
- ORB and SIFT modes reset reference frame when switching stabilization modes
- ECC refinement is implemented but currently disabled for performance reasons
- Isotropic scaling is automatically removed from motion estimates (may not suit all use cases)

## References

- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0-521-54051-3

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, bug reports, or feature requests.

---

**Note**: This implementation prioritizes real-time performance and educational value, demonstrating multiple approaches to video stabilization in a single, comprehensive system.
