# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_chessboard.png "Undistorted chessboard"
[image2]: ./examples/undistorted_road.png "Undistorted Road Transformed"
[image3]: ./examples/sobel_operator1.png "Sobel HLS threshold 1"
[image4]: ./examples/sobel_hls_threshold.png "Sobel HLS threshold 2"
[image5]: ./examples/birds_eye.png "Warp Example"
[image6]: ./examples/birds_eye_perspective.png "Birds Eye perspective Warp Example"
[image7]: ./examples/histogram.png "Histogram"
[image8]: ./examples/color_fit_lines.jpg "Fit Visual"
[image9]: ./examples/advanced_lane_lines_output.png "Output"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

A 3D object represented in a 2D image isn't prefect; that image is always distorted. We need to do Camera Calibration to figure out the magnitude of distortion in the image and we then need to correct that distortion. To figure out the distortion we use chessboards because it's contrast pattern makes it easy to detect and measure distortions.

I have used opencv function findChessboardCorners() to get the corners of all the 20 images in the camera_cal. We can get all the corners (my code only got corners of 17 of the 20 images) and create as set of points `objpoints` that maps to the corner point coordinates. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()`.

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

##### Camera Calibration

The first step in the pipeline is the camera calibration step described above. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The image and the video have a lot of other information like trees, sky, other cars. We need to filter out whatever's not needed which leaves out the open road and the lane lines. We use color and gradient thresholds to filter out what we don’t want.
We know some features of the lanes i.e. they are white or yellow; whereas the road is gray and that forms a high contrast between the two. The lane lines are amost vertical in the first two videos. Hence, we do a color threshold to figure out the lane lines like we did in the first assignement. We do a color threshold filter to pick only yellow and white elements, using opencv convert color to HSV space (Hue, Saturation and Value). The HSV dimension is suitable to do this, because it isolates color (hue), amount of color (saturation) and brightness (value). We define the range of yellow independent of the brightness i.e. in the shadow or bright sun we'll only catch the yellow lane line. Yellow lane lines are very useful as they are solid and highly visible. I have defined a threshold for the yellow and white lane lines.

```python
yellow_low = np.array([0,100,100])
yellow_high = np.array([50,255,255])

white_low = np.array([18,0,180])
white_high = np.array([255,80,255])
```

We use the Sobel operator to find the contrast. The sobel operator is nothing but a contrast filter that gives us the derivate between two neighbouring points. We use the x direction because the lanes are vertical and we need to find the contrast between the cement color of raod vs the yellow and white of the lane lines. Like any contrast filter the Solbel operator returns a high derivative (output) when the contrast is high. The steps of the Sobel Operator operation as follows.

1. Convert the undistorted image to grayscale.
2. Get the gradient using the `cv2.Sobel()` function for the channels L and S from HLS. This will give us a 0-255 range output for each pixel
2. Scale the ouput of sobel operator to `np.uint8()` by using `np.uint8(255*sobelx1/ np.max(sobelx1))`
3. Create a binary mask within the min and max threshold. This will give us the binary output that we desired.


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

##### Birds' Eye View

We need to warp the image such that it looks like its taken from above i.e. Birds Eye view. This is helpful in fitting in a curve because from the top the curve can be easy noticed whereas on the road it is straight lines we have to follow and turn when the turn.

We can use the cv2 functions `cv2.warpPerspective()` and  `cv2.getPerspectiveTransform()` to get the warped image.  The `cv2.getPerspectiveTransform()` function takes as inputs the source (`src`) and destination (`dst`) points .  I chose the hardcoded  source and destination points described below.  The `cv2.getPerspectiveTransform()` returns the transformation matrix (`M`) for the actually warping of the image. We then use `cv2.warpPerspective()` which takes in the actual image (`img`) and `M` and the image size (`img_size`) and returns the warped image i.e. the Bird's Eye view.

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

We can use the second order polynomial to fit the lane: x = ay**2 + by + c.

We use a histogram on the bottom half of image where the lane lines are; wherever the histogram spikes that's where our lane lines are.

![alt text][image7]
![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Refrenced from : https://www.intmath.com/applications-differentiation/8-radius-curvature.php

The curvature is important to know because then we'll know how much to deviate from the center to keep the car within the lane lines.

Taking the formulas straight out of the tutorial: https://www.intmath.com/applications-differentiation/8-radius-curvature.php

```radius_curvature=​​ (1 + (dy/dx)**2)**1.5 / abs(d2y /dx2)```

We can then calculate the radius of the left, right lanes and the base of the vehicle. After getting the radius we also need to translate the image pixels real world metres. I am using the lecture provided values.

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

```

Hence we get the radius of curvature and offset from center from the curves of the lines. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Once we know the position of lanes in birds-eye view and the radius of curvature, we use opencv function `cv2.polyfill()` to draw a area in the image.
Then, we warp back to original perspective, and merge it to the color image.

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I just ran the same code that I ran above for single images. A video is a set of images and it performed just like it did in the above images. I also added a validity check using the last position of the lane curvature as it won't differ drastically lane by lane so, I put in a threshold limit for it. The video result was good for the project_video.mp4 but for the challenge_video.mp4 and harder_challenge_video.mp4 it didn't work so well owing to the sun shadow and other distortions.

Here's a [link to my video result](./project_video_ouput.mp4)

Good results on challenge video [link to my challenge video result](./challenge_video_output.mp4)

Dreadful perfomance on [link to my challenge video result](./harder_challenge_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am happy that I could apply simple logic for the first project_video and the challenge_video and it worked. Picking up the yellow and white lanes and mapping that out. Being a software developer who hates hard-coding I should have made it worked for any color. I am also glad that cv2 functions were reaily available to me. My code is premature at the moment that I won't be able to use this in a real-wolrd scenario. As you can see in the harder challenge my algorithm really fails. The noise created by the tree shadow is very difficult to read. not to mention dissapearing lane lines. The first two videos I could cast my net on the wide open road but the last video I couldn't do that maybe I should have taken shorter samples of the image and work based off that. This algo won't work at all in night time. I really have no clue how to solve the harder challenge. 

To make it better I will add better threshold and validity check and take shorter samples of the road. I would also try to eradicate the shadows from the image that is caused y trees and the bridge above. I will also need to taken into account if both the lane lines are broken and white or both are yellow.
