---
layout: default
title: Mathematical Overview
---

## Mathematical Overview

This document provides a mathematical overview of the core concepts underlying the video stabilization system. It explains how camera motion is modeled and estimated using homography and more specifically rigid (Euclidean) transformations, and describes the decomposition of these transformations into fundamental geometric parameters. Understanding these principles is essential for grasping how the stabilizer separates and selectively suppresses different types of camera movement.

> **Note:** The following mathematical overview assumes that the reader is familiar with fundamental concepts in **Linear Algebra** (such as matrices, determinants, and matrix factorizations) and **Projective Geometry** (including homogeneous coordinates, projective transformations, and the geometric interpretation of homographies). A solid grasp of these topics is essential for understanding the derivations and parameterizations presented below.


### Homography Decomposition into fundamental parameters
The homography matrix is a central mathematical tool for modeling the geometric relationship 
between two images or views of an approximately planar scene(*) related by camera motion. To 
better understand and manipulate the effects of camera movement, it is useful to decompose the 
homography into a set of interpretable parameters, each corresponding to a fundamental 
geometric transformation such as rotation, scaling, shear, translation, and perspective 
distortion. This section introduces the mathematical framework for homography decomposition, 
outlines the meaning of each parameter, and explains how this decomposition enables 
fine-grained control over video stabilization and motion analysis.

(*A scene is considered "approximately planar or flat" if the differences in depth within the scene are much smaller than the distance from the camera to the scene.)

Below we cite (with small adaptations) from the popular, highly regarded *Multiple View Geometry in Computer Vision* book [1]:
> A projective transformation can be decomposed into a chain of transformations, where each matrix in the chain represents a transformation higher in the hierarchy than the previous one:  

$$
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
$$

>with $A$ a non-singular matrix given by $A = sRK+tv^\top$, and $K$ an upper-triangular matrix normalized as $detK = 1$. This decomposition is valid provided $\eta \neq 0$, and is unique if $s$ is chosen positive.
>
> Each of the matrices $H_S$, $H_A$, $H_P$ is the “essence” of a transformation of that type (as indicated by the subscripts $S$, $A$, $P$). [...] $H_P$ (2 dof) moves the line at infinity; $H_A$ (2 dof) affects the affine properties, but does not move the line at infinity; and finally, $H_S$ is a general similarity transformation (4 dof) which does not affect the affine or projective properties.

In typical video frame-to-frame registration with real camera motion, the bottom-right entry of the homography matrix, $\eta$, is non-zero. If $\eta$ becomes zero, this usually indicates a degenerate or pathological case in the motion estimation process, meaning the estimate has most likely failed. To handle these rare situations robustly, the codebase replaces the resulting homography with the identity matrix. Note that $\eta = 0$ most commonly arises during perspective removal—a scenario not addressed in our current work.

We further decompose the homography matrix into fundamental parameters, listed below, each corresponding to one of the elementary matrices described above:

- **Isotropic Scaling** ($s$): Isotropic scaling factor
- **Rotation** ($\theta$): In-image-plane rotation angle
- **Anisotropic Scaling** ($k_1, k_2$): Anisotropic scaling ratio where $k_1 \cdot k_2 = 1$
- **Shear** ($\delta$): Shear transformation value
- **Translation** ($t$): 2D translation vector $t=(t_x, t_y)$
- **Perspective** ($v$): Horizon line shift vector $v=(v_x, v_y)$

A homography transformation between two frames can be expressed in two equivalent forms: as a $3\times3$ matrix $H$ or using the parametric decomposition shown above. Both representations capture the same geometric transformation and can be uniquely converted between each other. Importantly, while the matrix form has $9$ entries, it effectively has $8$ degrees of freedom since homographies are defined up to an arbitrary scale factor – matching the $8$ scalar parameters in the decomposed form: 2 for isotropic scaling $s$ and rotation $θ$, 1 for anisotropic scaling ratio $k_1/k_2$, 1 for shear $\delta$, 2 for translation vector $(t_x, t_y)$, and 2 for perspective vector $(v_x, v_y)$.

The homography decomposition, as implemented in `decomposeHomography` function defined in [Stabilizer class](https://github.com/joao-gueifao-924/Video-Stabilization/blob/main/include/stabilizer.hpp), proceeds as follows:

1. **Normalization**: The input $3 \times 3$ homography matrix $H$ is normalized so that the bottom-right entry $h_{3,3}$ becomes $1$.

2. **Extraction of Components**:
   - The translation vector $t$ is extracted from the top-right $2 \times 1$ block of the normalized matrix.
   - The perspective vector $v$ is extracted from the bottom-left $1 \times 2$ block.
   - The upper-left $2 \times 2$ block ($A$) is also extracted.

3. **Projective Correction**:
   - The matrix $A$ is corrected for projective effects by subtracting the outer product of $t$ and $v$, yielding $sRK = A - t v^\top$.
   - This isolates the affine part of the transformation.

4. **Isotropic Scaling Extraction**:
   The determinant of the $(sRK)$ matrix is computed. Then, we make use of the following mathematical properties: 
   - For some scalar $s$ and a $n \times n$ square matrix $M$, we have $\det(s M) = s^n \det(M)$ 
   - $R$ is a $2 \times 2$ rotation matrix, hence has positive unitary determinant, $det(R) = 1$
   - $K$ is a $2 \times 2$  upper triangular matrix also with positive unitary determinant, $det(K)=1$

   The isotropic scaling factor $s$ is then given by  $s = \sqrt{\det(sRK)}$.

5. **QR Decomposition**:
   
   - *QR decomposition* is a mathematical technique that factors a matrix into two components: an orthogonal matrix $Q$ and an upper-triangular matrix $R$. (Note: in this paragraph only, $R$ refers to the upper-triangular matrix from the QR decomposition, not the rotation matrix $R$ used elsewhere in our decomposition — the names simply overlap.)

   - In our implementation, we apply QR decomposition (using the Gram-Schmidt process) to the matrix $(RK) = (sRK)/s$. This allows us to separate $(RK)$ into a true rotation matrix $R$ and an upper-triangular matrix $K$. 

   - Because we are working with $2 \times 2$ matrices, the classical Gram-Schmidt process is both simple and numerically stable enough for our needs, so we use it instead of more complex methods.

6. **Rotation angle, anisotropic scaling and shear parameters extraction**
    
   From the previous step, we obtained the rotation matrix $R$ and scaling matrix $K$:
   
   $$R = \begin{bmatrix}
   \cos(\theta) & -\sin(\theta) \\
   \sin(\theta) & \cos(\theta)
   \end{bmatrix}, \quad K = \begin{bmatrix}
   k_1 & \delta \\
   0 & k_2
   \end{bmatrix}$$
   
   We easily extract $\theta =\operatorname{atan2}(\operatorname{sin}(\theta), \operatorname{cos}(\theta))$ from the entries of $R$ and remaining parameters $k_1, k_2, \delta$ directly from the entries of $K$.

7. **Translation Correction**:
   - A similarity has exactly 1 fixed or invariant point, around which isotropic scaling and rotation take place. Normally, this fixed point is the coordinate system origin $O=0$.
   - The general form for a similarity with an arbitrary fixed point $c$, is:

    $$
    \begin{align*}
    p' &=& sR(p-c) + c + t = \\
       &=& sRp - sRc + c + t = \\
       &=& sRp + t + c - sRc = \\
       &=& sRp + [t + (I-sR)c] = \\
       &=& sRp + t^+ \\
    \end{align*}
    $$
    
   - The translation vector $t^+$ captures both camera translational movement $t$ and the additional shift resulting from scaling and rotating around the point $c$.
   - In OpenCV and other image processing and computer vision libraries, the coordinate system origin $O=0$ is defined at the top-left corner of the image. However, for the purposes of camera motion stabilization, we choose to model rotations around the camera's optical axis, as this better approximates the typical axis of rotational shake (e.g., when someone films handheld with a cellphone). In practice, for typical cameras, the optical axis is well-approximated by the image center, with coordinates $(\frac{W}{2}, \frac{H}{2})$, where $W$ and $H$ denote the image width and height, respectively. For this reason, we set $c$ to be the image center. (Note: the true rotation axis can be anywhere, and in the given example of handheld filming with a cellphone, the rotation axis almost never is the optical axis, since the camera lenses are frequently located at the corners of the device. Nonetheless, for our practical purposes, this approximation is sufficient.)
   - To center the homography decomposition at $c$, we adjust the translation as $t = t^+ - (I - sR)c$ to remove scaling and rotation effects about $c$.

This decomposition allows the stabilizer to independently manipulate rotation, translation, scaling, shear, and perspective components of the camera motion, enabling fine-grained stabilization modes.  For implementation reference, see the `decomposeHomography` and its inverse `composeHomography` functions defined in [Stabilizer class](https://github.com/joao-gueifao-924/Video-Stabilization/blob/main/include/stabilizer.hpp).


### Modelling Camera Motion

In this section, we introduce the mathematical models used to describe camera motion between video frames. Understanding these models is essential for analyzing and stabilizing video sequences, as they provide a framework for representing how the camera moves and how this motion affects the appearance of the captured images.


Let $I_t(p)$ denote the image captured by a standard 8-bit color camera at time $t \in \mathbb{Z}$. Here, $p = (x, y) \in \mathbb{Z}^2$ specifies the pixel coordinates, and $I_t(p): \mathbb{Z}^2 \to \lbrace0,1,\dots,255\rbrace^3$ returns the RGB color value at each pixel.

In image $I_{t+k}$, for $k \in \mathbb{Z}$, i.e., at an earlier or later time instant $t+k$, we perceive same visual contents but in different locations due to camera motion:

$$
I_{t+k}(p_{t+k}) \approx I_{t}(p_{t})
$$

Here, $\approx$ indicates that the two images are only approximately equal. This is because imaging noise, as well as changes in the scene’s structure or lighting, can occur between times $t$ and $t+k$, causing slight differences between the images. For this approximation to be valid, the time gap $k$ should be small to minimize these effects. Furthermore, any changes in the scene—such as moving objects, shape changes, or new objects entering or leaving the frame—should be relatively minor compared to the overall image size. In summary, the two images must retain enough visual similarity to be reliably aligned.

To describe camera motion between two time points, we can use a homography transformation:

$$
p_{t+k} \sim {^{t+k}H}_t\, p_t
$$

Here, $p_{t+k}$ represents the position of a point in the image at time $t+k$, and ${^{t+k}H}_t$ is the homography matrix that maps the point $p_t$ from time $t$ to its new location at time $t+k$. Warping image $I_t$ using ${^{t+k}H}_t$ will make it aligned with image $I_{t+k}$. 

It's important to note that using a homography to model camera motion is an approximation. This method works best when the scene is relatively flat or the camera is far from the scene. If the camera is close to objects with varying depths, the homography may not accurately capture the true motion, which can limit the effectiveness of video stabilization.

A homography transformation has 8 degrees of freedom, allowing it to represent a wide range of image motions. However, this flexibility can lead to overfitting, especially when the transformation is estimated from noisy/inaccurate feature point matches.

To improve robustness, we can use a simpler model: the 2D rigid transformation: 

$$
p_{t+k} \sim {^{t+k}T}_t\, p = \begin{bmatrix} {^{t+k}R}_t & {^{t+k}t}_t \\0^\top & 1\end{bmatrix}
$$

where ${^{t+k}R}_t$ is a $2 \times 2$ rotation matrix and $ {^{t+k}t}_t$ is a $2 \times 1$ translation vector. They represent motion exclusively on the image plane, from time instant $t$ to $t+k$.

This model has only 3 degrees of freedom—horizontal and vertical translation and in-plane rotation—which directly correspond to typical camera movements on the image plane. By reducing the number of parameters, the 2D rigid model is less sensitive to noise and produces more stable results.

However, this simpler model also has its drawbacks. When the camera undergoes more complex movements—such as zooming or significant tilts and rotations out of the image plane—the 2D rigid transformation cannot accurately capture these motions. As a result, the video stabilization process, which relies on an appropriate motion model, may produce visible motion artifacts in the output video.

The 2D rigid (Euclidean) transformation is simply a specific case of the broader 2D homography. In this document, we use the term "homography" to describe all transformations between video frames—whether they are rigid (rotation and translation only) or more general (including scaling, shear, or perspective). When needed, we will clearly indicate the specific type of transformation being discussed.

#### Transformation chaining

Between successive video frames, the tranformations that align consecutive pairs of images can be chained. We start with mapping points in image $I_t$ to corresponding points in older images $I_{t-1}, I_{t-2}, I_{t-3}, \dots$:

$$
\begin{align*}
p_{t-1} &\sim& {^{t-1}H}_t\, p_t \\
p_{t-2} &\sim& {^{t-2}H}_{t-1}\, p_{t-1} &\sim& {^{t-2}H}_{t-1}\, {^{t-1}H}_t\, p_t \\
p_{t-3} &\sim& {^{t-3}H}_{t-2}\, p_{t-2} &\sim& {^{t-3}H}_{t-2}\, {^{t-2}H}_{t-1}\, {^{t-1}H}_t\, p_t \\
\vdots
\end{align*}
$$
Generalizing for $t-l$, with $l \in \mathbb{Z}_{>0}$:

$$
\begin{align*}
p_{t-l} &\sim {^{t-l}H}_{t-l+1}\,{^{t-l+1}H}_{t-l+2} \cdots {^{t-1}H}_t\, p_t \Leftrightarrow \\

\Leftrightarrow p_{t-l} &\sim \overset{\curvearrowleft}{\prod_{i=1}^{l}}\left( {^{t-i}H}_{t-i+1} \right) p_t
\end{align*}
$$

Where $$\overset{\curvearrowleft}{{\prod_{i=1}^{N}}}\left( M_i \right) = M_N . M_{N-1}.M_{N-2}. \cdots . M_1$$ denotes the product series of matrices that expands to the left (matrix product is not commutative—hence we must specify its order).

We now proceed in a similar manner for mapping points in image $I_t$ to corresponding points in future images $I_{t+1}, I_{t+2}, I_{t+3}, \dots$:


$$
\begin{align*}
p_{t+1} &\sim& {^{t+1}H}_t\, p_t \\
p_{t+2} &\sim& {^{t+2}H}_{t+1}\, p_{t+1} &\sim& {^{t+2}H}_{t+1}\, {^{t+1}H}_t\, p_t \\
p_{t+3} &\sim& {^{t+3}H}_{t+2}\, p_{t+2} &\sim& {^{t+3}H}_{t+2}\, {^{t+2}H}_{t+1}\, {^{t+1}H}_t\, p_t \\
\vdots
\end{align*}
$$
Generalizing for $t+r$, with $r \in \mathbb{Z}_{>0}$:

$$
\begin{align*}
p_{t+r} &\sim {^{t+r}H}_{t+r-1}\,{^{t+r-1}H}_{t+r-2} \cdots {^{t+1}H}_t\, p_t \Leftrightarrow \\

\Leftrightarrow p_{t+r} &\sim \overset{\curvearrowleft}{\prod_{i=1}^{r}}\left( {^{t+i}H}_{t+i-1} \right) p_t
\end{align*}
$$

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
- Uses **2D Rigid (Euclidean)** motion model (rotation + translation, no scaling)
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
- **Motion Models**: Rigid (Euclidean) transformation fitting
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhanced feature detection
- **QR Decomposition**: Gram-Schmidt process for 2×2 matrix factorization

## References

- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0-521-54051-3
