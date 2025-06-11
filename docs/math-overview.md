---
layout: default
title: Mathematical Overview
---

# Mathematical Foundations for Real-Time Video Stabilization

This document provides a mathematical overview of the core concepts underlying a video stabilization system ("stabilizer"), available on [GitHub](https://github.com/joao-gueifao-924/Video-Stabilization). It explains how camera motion is modeled and estimated using homographies and, more specifically, rigid (Euclidean) transformations, and describes the decomposition of these transformations into fundamental geometric parameters. In addition, we show how to smooth out shaky video by averaging the estimated inter-frame camera motions, a technique that reduces unwanted shake while preserving intentional movement. Understanding these principles is essential for grasping how the stabilizer separates and selectively suppresses different types of camera movement.

The following mathematical overview assumes that the reader is familiar with fundamental concepts in **Calculus** and **Linear Algebra** (such as summation and product series, matrices, determinants, and matrix factorizations) and **Projective Geometry** (including homogeneous coordinates, projective transformations, and the geometric interpretation of homographies). A solid grasp of these topics is essential for understanding the derivations and parameterizations presented below.


## Homography Decomposition into fundamental parameters
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

In typical video frame-to-frame registration with real camera motion, the bottom-right entry of the homography matrix, $\eta$, is non-zero. If $\eta$ becomes zero, this usually indicates a degenerate or pathological case in the motion estimation process, meaning the estimate has most likely failed. To handle these rare situations robustly, the stabilizer codebase replaces the resulting homography with the identity matrix. Note that $\eta = 0$ most commonly arises during perspective removal—a scenario not addressed in our current work.

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
    p' &=& sR(p-c) + c + t \\
       &=& sRp - sRc + c + t \\
       &=& sRp + t + c - sRc \\
       &=& sRp + [t + (I-sR)c] \\
       &=& sRp + t^+ \\
    \end{align*}
    $$
    
   - The translation vector $t^+$ captures both camera translational movement $t$ and the additional shift resulting from scaling and rotating around the point $c$.
   - In OpenCV and other image processing and computer vision libraries, the coordinate system origin $O=0$ is defined at the top-left corner of the image. However, for the purposes of camera motion stabilization, we choose to model rotations around the camera's optical axis, as this better approximates the typical axis of rotational shake (e.g., when someone films handheld with a cellphone). In practice, for typical cameras, the optical axis is well-approximated by the image center, with coordinates $(\frac{W}{2}, \frac{H}{2})$, where $W$ and $H$ denote the image width and height, respectively. For this reason, we set $c$ to be the image center. (Note: the true rotation axis can be anywhere, and in the given example of handheld filming with a cellphone, the rotation axis almost never is the optical axis, since the camera lenses are frequently located at the corners of the device. Nonetheless, for our practical purposes, this approximation is sufficient.)
   - To center the homography decomposition at $c$, we adjust the translation as $t = t^+ - (I - sR)c$ to remove scaling and rotation effects about $c$.

This decomposition allows the stabilizer to independently manipulate rotation, translation, scaling, shear, and perspective components of the camera motion, enabling fine-grained stabilization modes.  For implementation reference, see the `decomposeHomography` and its inverse `composeHomography` functions defined in [Stabilizer class](https://github.com/joao-gueifao-924/Video-Stabilization/blob/main/include/stabilizer.hpp).


## Modelling Camera Motion

In this section, we introduce the mathematical models used to describe camera motion between video frames. Understanding these models is essential for analyzing and stabilizing video sequences, as they provide a framework for representing how the camera moves and how this motion affects the appearance of the captured images.


Let $I_t(p)$ represent the image captured at time $t \in \mathbb{Z}$ by a standard camera with rectilinear (non-distorting) lenses, following the pinhole camera model. This video stabilization system is specifically designed for such cameras. It is not intended for use with other camera or lens types—such as omnidirectional or fisheye lenses—which are outside the scope of this work and may not function correctly with this system.

Here, $p = (x, y) \in \mathbb{Z}^2$ represents the pixel coordinates, and $I_t(p): \mathbb{Z}^2 \to C^n$ gives the color intensity at each pixel. The value of $n$ depends on the type of camera: $n=1$ for monochrome (grayscale) images, and $n>1$ for color images—most commonly, $n=3$ for RGB. Typically, each color channel uses 8 bits, so $C = \{0, 1, \dots, 255\}$. 

The stabilizer itself does not depend on the specific image format. Instead, it relies on external algorithms for finding sparse feature point correspondences, and abstracts away the details of how image data is represented.

In image $I_{t+k}$, for $k \in \mathbb{Z}$, i.e., at an earlier or later time instant $t+k$, we perceive same visual contents but in different locations due to camera motion:

$$
I_{t+k}(p_{t+k}) \approx I_{t}(p_{t})
$$

Here, $\approx$ indicates that the two images are only approximately equal. This is because imaging noise, as well as changes in the scene’s structure or lighting, can occur between times $t$ and $t+k$, causing slight differences between the images. For this approximation to be valid, the time gap $k$ should be small to minimize these effects. Furthermore, any changes in the scene—such as moving objects, shape changes, or new objects entering or leaving the frame—should be relatively minor compared to the overall image size. In summary, the two images must retain enough visual similarity to be reliably aligned.

To describe camera motion between two time points, we can use a homography transformation:

$$
p_{t+k} \sim {^{t+k}H}_t\, p_t
$$

Here, $p_{t+k}$ represents the position of a point in the image at time $t+k$, and ${^{t+k}H_t}$ is the homography matrix that maps the point $p_t$ from time $t$ to its new location at time $t+k$. Warping image $I_t$ using ${^{t+k}H_t}$ will make it aligned with image $I_{t+k}$.

It's important to note that using a homography to model camera motion is an approximation. This method works best when either:
- the scene is relatively flat
- the camera is far from the scene
- the camera undergoes a pure rotation (i.e., no translation)

When the camera moves (translates) while being close to objects at different depths, a homography may not accurately describe the true motion. This limitation can reduce the effectiveness of video stabilization. Homography is a global, parametric motion model—it assumes the same transformation applies everywhere in the image. In contrast, other motion models, such as non-parametric models and non-rigid registration, can better capture camera motion against the scene, especially in scenes with significant depth variation compared to camera distance.

Despite these limitations, homographies (and their special cases, like rigid or affine transformations) are widely used for video stabilization because they provide a powerful albeit simple and intuitive framework based on linear algebra.

A homography transformation has 8 degrees of freedom, allowing it to represent a wide range of image motions. However, this flexibility can lead to overfitting, especially when the transformation is estimated from noisy/inaccurate feature point matches.

To improve robustness, we can use a simpler model: the 2D rigid transformation: 

$$
p_{t+k} \sim {^{t+k}T}_t\, p = \begin{bmatrix} {^{t+k}R}_t & {^{t+k}t}_t \\0^\top & 1\end{bmatrix}
$$

where ${^{t+k}R}_t$ is a $2 \times 2$ rotation matrix and $ {^{t+k}t}_t$ is a $2 \times 1$ translation vector. They represent motion exclusively on the image plane, from time instant $t$ to $t+k$.

This model has only 3 degrees of freedom—horizontal and vertical translation and in-plane rotation—which directly correspond to typical camera movements on the image plane. By reducing the number of parameters, the 2D rigid model is less sensitive to noise and produces more stable results.

However, this simpler model also has its drawbacks. When the camera undergoes other types of movement—such as zooming or significant tilts and rotations out of the image plane—the 2D rigid transformation cannot accurately capture these motions. As a result, the video stabilization process, which relies on an appropriate motion model, may produce visible motion artifacts in the output video.

The 2D rigid (Euclidean) transformation is simply a special case of the broader 2D homography. In this document, we use the term "homography" to describe all transformations between video frames—whether they are rigid (rotation and translation only) or more general (including scaling, shear, or perspective). When needed, we will clearly indicate the special type of transformation being discussed.

### Transformation chaining

Transformations that align consecutive video frames can be combined—this process is called "chaining." By chaining transformations, we can map points from one frame to their corresponding locations in earlier or later frames in the video.

This is possible because our motion model is global and parametric: each transformation can be represented as a matrix, and we can use standard matrix multiplication to combine them. When we multiply two homography matrices, the result is another homography, making it easy to express complex motion as a sequence of simpler steps.

For example, to find where a point in image $I_t$ came from in previous frames $I_{t-1}, I_{t-2}, I_{t-3}, \dots$, we multiply the appropriate homography matrices together:

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

Where

$$\overset{\curvearrowleft}{\prod_{i=1}^{N}} \left( M_i \right) = M_N . M_{N-1}.M_{N-2}. \cdots . M_1$$

denotes the product series of matrices that expands to the left (matrix product is not commutative—hence we must specify its order).

We now proceed in a similar manner for mapping points in image $I_t$ to their corresponding locations in future images $I_{t+1}, I_{t+2}, I_{t+3}, \dots$:

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

## Smoothing camera motion

This section explains the mathematical reasoning behind the `calculateGlobalSmoothingStabilization` function, as implemented in the [Stabilizer class](https://github.com/joao-gueifao-924/Video-Stabilization/blob/main/include/stabilizer.hpp). It clarifies how and why the algorithm smooths camera motion by averaging transformations across a temporal window.

To smooth camera motion, we apply a low-pass filter over the sequence of values for $p_t$, for varying $t$. A suitable candidate for such filter is the moving average, where we substitute the value for each $p_t$ by the average of the values within its temporal neighbourhood:

$$
\begin{align*}
\bar{p_t} &= \frac{p_{t-M} + p_{t-(M-1)} + \cdots + p_{t-1} + p_t + p_{t+1} + \cdots + p_{t+(N-1)} + p_{t+N}}{M+1+N} \\
&= \frac{1}{M+1+N} \left( \sum_{l=+1}^{+M}p_{t-l} + p_t + \sum_{r=+1}^{+N}p_{t+r} \right)
\end{align*}
$$

where $M$ and $N$ are the number of video frames to the left and right of the presented frame at time $t$, respectively.

We can now express both $p_{t-l}$ and $p_{t+r}$ in terms of $p_t$ by applying the appropriate homography transformations (we change  equality, $=$, by proportionality, $\sim$):

$$
\bar{p_t} \sim \frac{1}{M+1+N} \left( \sum_{l=+1}^{+M} \overset{\curvearrowleft}{\prod_{i=1}^{l}}\left( {^{t-i}H}_{t-i+1} \right) p_t + p_t + \sum_{r=+1}^{+N} \overset{\curvearrowleft}{\prod_{i=1}^{r}}\left( {^{t+i}H}_{t+i-1} \right) p_t \right)
$$

We now factor out $p_t$:

$$
\begin{align*}
\bar{p_t} &\sim \frac{1}{M+1+N} \left( \sum_{l=+1}^{+M} \overset{\curvearrowleft}{\prod_{i=1}^{l}}\left( {^{t-i}H}_{t-i+1} \right) + I + \sum_{r=+1}^{+N} \overset{\curvearrowleft}{\prod_{i=1}^{r}}\left( {^{t+i}H}_{t+i-1} \right) \right) p_t \\
&\sim Q_t\, p_t
\end{align*}
$$

where $I$ denotes the $3 \times 3$ identity matrix.

To reduce camera shake, we stabilize each frame $I_t$ by warping it with the homography $Q_t$. This smoothing homography $Q_t$ is calculated by combining the sequence of frame-to-frame homographies between neighboring frames, using the formula shown above. These individual homographies can be estimated in several ways; a common and effective method is to track sparse feature points between frames and use their correspondences to compute the transformations, as we'll see in next section.

The choice of values for $M$ and $N$ presents a trade-off. While increasing both past and future window sides yields a smoother stabilization, it comes at the expense of increased image cropping when in presence of strong camera shake. Also, a larger future window side yields an increased video presentation delay, as all frames ahead of the presented frame $I_t$ will need to be buffered so that computations can take place.


## Image registration

When estimating camera motion between consecutive video frames, optical flow is a well-suited technique. Optical flow relies on the brightness constancy assumption, which states that the intensity of a point in the image remains constant as it moves from one frame to the next. For a detailed explanation of the brightness constancy model and optical flow algorithms, readers are encouraged to consult standard references on the topic.

To accomplish image registration, we rely on robust and efficient algorithm implementations provided by OpenCV. The process consists of two main steps:

1. **Feature Point Detection:**  
   For each video frame, we identify strong, trackable points using the Shi-Tomasi corner detector, available in OpenCV as [cv::goodFeaturesToTrack](https://docs.opencv.org/4.11.0/d4/d8c/tutorial_py_shi_tomasi.html).

2. **Feature Point Tracking:**  
   Once these feature points are detected, we track their movement from one frame to the next using the Lucas-Kanade Pyramidal Optical Flow algorithm, implemented as [cv::calcOpticalFlowPyrLK](https://docs.opencv.org/4.11.0/d4/dee/tutorial_optical_flow.html). This method computes a sparse optical flow, efficiently following the detected points across consecutive frames.
3. **Estimate the transformation between frames:**  
   For each pair of consecutive video frames, we use the matched feature points to compute a transformation that best aligns them. Instead of fitting a general homography, we estimate a rigid body motion. This model includes only rotation and translation. 

   Fitting a full homography can easily overfit to inaccurate point correspondences, even when using robust estimation methods like RANSAC. Overfitting can introduce unwanted distortions and instability. Additionally, if the scene is not approximately planar (for example, if there are objects at significantly different depths compared to camera distance), the estimated homography may be unreliable, further degrading stabilization quality. By focusing on rigid motion, we minimize the effect of these innacuracies and achieve more reliable video stabilization.

   To find this transformation, we determine the optimal mapping from the detected points in one frame to their corresponding points in the next. We use OpenCV’s [cv::estimateAffinePartial2D](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#gad767faff73e9cbd8b9d92b955b50062d) function for this purpose. Despite its name, this function actually computes a similarity—composed of uniform scaling, rotation and translation, but not anisotropic scaling nor shear. The term "partial affine" can be misleading, as it suggests a more general transformation than what is actually computed.

   Note that OpenCV does not provide a direct method to fit a pure rigid motion to point correspondences. As a workaround, we compute a similarity transformation (which may include uniform scaling) and then explicitly remove any scaling component, ensuring the result is a true rigid transformation. This provides good enough results for our video stabilization needs.

   If your imaging system is accurate enough (for example, if there is no motion blur), you can choose to use a full homography transformation instead of a rigid or similarity transform. This substitution is straightforward—see the `estimateMotion` function in the [Stabilizer class](https://github.com/joao-gueifao-924/Video-Stabilization/blob/main/include/stabilizer.hpp) for details on how to implement this change.

## Motion Cancellation

### Stop All Camera Motion (Freeze the View)

Instead of merely smoothing out camera motion, we can completely eliminate it—making the scene appear perfectly still, as if the camera were locked in place. For this to happen, each image $I_t$ is warped using the homography matrix that aligns it to image $I_{t-L}$, i.e., a reference frame captured $L$ frames ago:

$$
p_{t-L} \sim {^{t-L}H}_t\, p_t
$$

This results in all camera movement being canceled, and the scene remaining visually fixed on the display.

To compute this homography matrix, we have implemented three different approaches:

- **Accumulated Optical Flow**: This matrix is constructed by chaining together all inter-frame transformations from a chosen reference (or anchor) frame up to the current frame, and then taking the inverse of this combined transformation. Inaccuracies in estimating the motion between frames—especially those caused by bland scenes (few trackable feature points), motion blur (from a slow shutter speed relative to camera movement), or uncorrected lens distortion—tend to accumulate as the transformations are chained together. As a result, the supposedly static scene may gradually drift over time. User may want to update the reference frame from time to time to counteract this issue.

- **ORB-based Registration**: Employs Oriented FAST and Rotated BRIEF features for direct frame alignment between presented frames and a reference frame. Registering or aligning the current frame directly into the reference frame avoids the gradual drift which occurs due to innacurate inter-frame motion accumulation. Unfortunately, the produced results were poor, at least for the camera and hyperparameters used while development of the stabilizer took place. Further research is deemed necessary.
- **SIFT-based Registration**: Same as for ORB-based Registration but uses Scale-Invariant Feature Transform instead for highest accuracy in frame registration. Unfortunately, the results were also poor. Further research is deemed necessary.

# Note: 
**all content below is still a work in progress**

### Cancel Camera Rotation
TODO. Kills all camera rotation, only allowing translational camera motion to take place.

### Cancel Camera Translation
TODO. Not deemed very useful in practive, but still good for educational purposes.

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

## References

- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. ISBN: 978-0-521-54051-3
