
## Mathematical Overview

This document provides a mathematical overview of the core concepts underlying the video stabilization system. It explains how camera motion is modeled and estimated using homography and more specifically rigid-body transformations, and describes the decomposition of these transformations into fundamental geometric parameters. Understanding these principles is essential for grasping how the stabilizer separates and selectively suppresses different types of camera movement.

> **Note:** The following mathematical overview assumes that the reader is familiar with fundamental concepts in **Linear Algebra** (such as matrices, determinants, and matrix factorizations) and **Projective Geometry** (including homogeneous coordinates, projective transformations, and the geometric interpretation of homographies). A solid grasp of these topics is essential for understanding the derivations and parameterizations presented below.


### Homography Decomposition into fundamental parameters

Below we cite (with small adaptations) from the popular, highly regarded *Multiple View Geometry in Computer Vision* book [1]:
> A projective transformation can be decomposed into a chain of transformations, where each matrix in the chain represents a transformation higher in the hierarchy than the previous one:  

```math
H = H_S H_A H_P =
\begin{matrix}
sR & t \\
0^\top & 1
\end{matrix}
\begin{bmatrix}
K & 0 \\
0^\top & 1
\end{matrix}
\begin{matrix}
I & 0 \\
v^\top & \eta
\end{matrix}
=
\begin{matrix}
A & t \\
v^\top & \eta
\end{matrix}
```

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

1. **Normalization**: The input $3 \times 3$ homography matrix $H$ is normalized so that the bottom-right entry $h_{3,3}$ becomes $1$.

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

## Technical References

- **Lucas-Kanade Optical Flow**: Shi-Tomasi corner detection with pyramidal LK tracking
- **ORB Features**: Oriented FAST keypoints with rotated BRIEF descriptors  
- **SIFT Features**: Lowe's Scale-Invariant Feature Transform algorithm
- **Homography Estimation**: RANSAC-based robust parameter estimation
- **Motion Models**: Euclidean (rigid-body) and affine transformation fitting
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhanced feature detection
- **QR Decomposition**: Gram-Schmidt process for 2×2 matrix factorization

## References

- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0-521-54051-3
