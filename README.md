# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


# Advanced Lane Finding Project
---

## The goals / steps of this project were as following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## * Camera Calibration
#### 1. Have the camera matrix and distortion coefficients been computed correctly and checked on one of the calibration images as a test?

The code for camera calibration is contained in the cells 2-7 of the IPython notebook(Project_Advanced_lane_Finding.ipynb).

I start by looking at the sample image, print the size of the image which happens to be (720, 1280, 3).
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/Sample_img1.png)


Since the chess board has 9 corners in a row and 6 corners in a column , the "object points", will be the (9, 6, 0) . Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are
the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints
will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
imgpoints will be appended with the (9,6) pixel position of each of the corners in the image plane with
each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion
coefficients using the cv2.calibrateCamera() function as shown below in the code.

```<python>

nx = 9
ny = 6

objpoints = []
imgpoints = []

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

fnames = glob.glob("camera_cal/calibration*.jpg")

for fname in fnames:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[:2],None,None)      

```


## * Distortion Correction
#### use the object and image points to caliberate the camera and compute the camera matrix and distortion coefficients
The code for creating distortion correction is contained in the cell 9 of the IPython notebook(Project_Advanced_lane_Finding.ipynb).
```<python>
#We use the cameraMatrix and distortionCoeffs to undistort the image.
img = mpimg.imread('camera_cal/calibration1.jpg')
undistorted = cv2.undistort(img, cameraMatrix, distortionCoeffs, None, cameraMatrix)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```

The result of undistorted image after applying distortion correction to the image using cv2.undistort is as below:-
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/sample_img2.png)
 
## *create a thresholded binary image using color transforms and gradients. 
The 2 kinds of gradient thresholds I applied was as below:
  1.Along the X axis.
  2.Directional gradient with thresholds of 30 and 90 degrees.
The following color thresholds is used in project:
   1.R & G channel thresholds so that yellow lanes are detected well.
   2.L channel threshold so that we don't take into account edges generated due to shadows.
   3.S channel threshold since it does a good job of separating out white & yellow lanes.
 The code for creating a threshold binary image is contained in the cells 11-12 of the IPython notebook(Project_Advanced_lane_Finding.ipynb).
The result of undistorted image after applying gradients and color thresholds to the image is as below:-
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/threshold_image.png)
   
 ## *Apply a perspective transform to rectify binary image ("birds-eye view").  
 The code for applying perspective transform is contained in the cells 16 of the IPython notebook(Project_Advanced_lane_Finding.ipynb)
```<python>
# Vertices extracted manually for performing a perspective transform
bottom_left = [220,720]
bottom_right = [1110, 720]
top_left = [570, 470]
top_right = [722, 470]

source = np.float32([bottom_left,bottom_right,top_right,top_left])

pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
pts = pts.reshape((-1,1,2))
copy = img.copy()
cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]

dst = np.float32([bottom_left,bottom_right,top_right,top_left])
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (image_shape[1], image_shape[0])

warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
    
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(copy)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```
Below is the result of image after applying perspective transform:- 
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/warped_image.png)

This resulted in the following source and destination points:

Source|	Destination
-------- |--------
220,720 |320,720
1110,720|920, 720
570, 470|320, 1
722, 470|920, 1

I verified that my perspective transform was working as expected by drawing the src and dst points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

## *Detect lane pixels and fit to find the lane boundary.
I plotted Histogram,The peaks in the histogram tell us about the likely position of the lanes in the image
```<python>
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

# Peak in the first half indicates the likely position of the left lane
half_width = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:half_width])

# Peak in the second half indicates the likely position of the right lane
rightx_base = np.argmax(histogram[half_width:]) + half_width

print(leftx_base, rightx_base)
plt.plot(histogram)
```
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/histogram.png)

The code for performing a sliding window search are in cell 22 of  the IPython notebook(Project_Advanced_lane_Finding.ipynb), starting with the base likely positions of the 2 lanes, calculated from the histogram.I have used 10 windows of width 100 pixels.The x & y coordinates of non zeros pixels are found, a polynomial is fit for these coordinates and the lane lines are drawn. 
The code for performing Searching around a previously detected line is in cell 23.Since consecutive frames are likely to have lane lines in roughly similar positions, in this section we search around a margin of 50 pixels of the previously detected lane lines.
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/Lane_pixels2.png)
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/lane_lines.png)

## *Determine the curvature of the lane and vehicle position with respect to center.
The code for finding the radius of curvature is in cell 24 of the IPython notebook(Project_Advanced_lane_Finding.ipynb).
The radius of curvature is computed according to the formula and method described in the tutorial.
Since we perform the polynomial fit in pixels but the curvature had to be calculated in real world meters,
we have to use a pixel to meter transformation and recompute the fit again.
The mean of the lane pixels closest to the car gives us the center of the lane. The center of the image gives us the
position of the car. The difference between the 2 is the offset from the center.
The result after running the cell is as follows:-

Radius of curvature| 4709.52 m
---------------- |----------------
Center offset| 0.14 m

## *Warp the detected lane boundaries back onto the original image.
I perform Inverse Transform 
  1.Paint the lane area
  2.Perform an inverse perspective transform
  3.Combine the precessed image with the original image.
  The code for warpping the detected lane boundaries back onto the orignial image is in cell 25 of the IPython notebook(Project_Advanced_lane_Finding.ipynb).
  Result after running cell 25 is as below:-
 ![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/final_pipeLine_processed_image2.png)
 ![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/ImagesOfProject/final_pipeLine_processed_image.png)

## *Final Pipeline video is as below:-
![alt tag](https://github.com/GirijaB/CarND-Advanced-Lane-Lines/blob/master/project_video_output.mp4)

## Discussions
## *Issues and Challenges
The video pipeline developed in this project did a fairly good job of detecting the lane lines in the test video provided for the project, which shows a road in basically ideal conditions, with fairly distinct lane lines, and on a clear day though it did lose the lane lines momentarily when there was heavy shadow over the road under a bridge.

challenging in this project is is finding a single resolution which produces the same quality result in any weather condition. I have not yet tested the pipeline on with varying lighting and weather conditions like rainy, snow, fog. and also at various road quality conditions where there is some road constructions going on, in city limits,or if there are faded lane lines or dark tire marks , or cracks and other lines that can create false lane markers on road. 



