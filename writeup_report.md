# Vehicle Detection and Tracking Project
 
## Introduction

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected


## Rubric Points
---

### Files Submitted

#### 1. Submission includes all required files and can be used to obtain the final video.

My project includes the following files:
* **P5-VehicleDetectionAndTracking.ipynb** contains the python code and in the following writeup all references to python code points to this file
* **writeup_report.md** summarizes the process used and results
* **output_images** directory contains output images shown in this writeup
* The final output video is on YouTube and linked at the end of this report

### Writeup/README

#### 1. Provide a Writeup/README that includes all the rubric points and how you addressed each one.

You're reading the Writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images and why?

I started by reading in all the `car` and `non-car` images. Here is an example of one of each:

![alt text](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking/blob/master/output_images/ExampleCarNoncar.jpg)

The code for this step is contained in the 2nd code cell of the IPython notebook (**P5-VehicleDetectionAndTracking.ipynb**). As an aside, the images are in the .png format. I'll be using cv2 function for reading the images which is in the BGR format. However, for plotting with `matplotlib.image` which uses RGB format, I'll convert the images to RGB from BGR format. 

There are 8792 car and 8968 non-car images. As these are roughly equal in number I don't see class imbalance issues with this data set.

Next I see how the Histogram of Oriented Gradients (HOG) for a car image looks like. Here I use HOG parameters of `color space=HSV`,  `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The HOG features are extracted by importing the `hog` module from the `skimage` package. The code is shown in cells 3 and 4 in the IPython notebook. The following picture shows the original car picture along with its hog features for each of the color channels `H`, `S`, and `V`. HOG features represent a distinct signature for the original image and they are preferred over other features such as using color transforms because they are robust for variations in the image.

![alt text](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking/blob/master/output_images/HOG%20Features.jpg)

#### 2. Explain how you settled on your final choice of HOG parameters.

As you see there are four HOG parameters and I'll now explore them to select the optimum parameters which will help me extract the best HOG features from the image. This process involved fixing (control features) three of these features to reasonable values and varying (experiment feature) the fourth feature and running the SVM classifier to see which one of the experiment option produced the best SVM classifier test accuracy. The SVM classifier is discussed further down. 

I first explored with three options for pixels per cell (`pixels_per_cell=[(8, 8), (16, 16), (32, 32)]` ) and fixing the other HOG parameters to `color_space = HSV`, `orientations=9` and `cells_per_block=(2, 2)`. This is in cell 5 of the notebook. I got the following results with highest test accuracy for `8x8` pixels per cell:

    Pixels per cell 8
    Test Accuracy of SVC =  0.9842
    -------------------------------
    Pixels per cell 16
    Test Accuracy of SVC =  0.9823
    -------------------------------
    Pixels per cell 32
    Test Accuracy of SVC =  0.9817
    -------------------------------
    
Next I explored two options for cells per block (`cells_per_block=[(2, 2), (4, 4)]`) and fixing the other HOG parameters to `color_space = HSV`, `orientations=9` and `pixels_per_cell=(8, 8)`. This is in cell 6 of the notebook. I got the following results with highest test accuracy for `2x2` cells per block:

    Cells per block 2
    Test Accuracy of SVC =  0.9899
    ------------------------------
    Cells per block 4
    Test Accuracy of SVC =  0.9879
    ------------------------------
    
Next I explored varying six color spaces (`color_space = [RGB, HSV, HLS, YCrCb, YUV, LUV]`) and fixing the other HOG parameters to `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The code is shown in cell 7 of the notebook. I got the following results with highest test accuracy for `YCrCb` color space:

    Color space:  RGB
    Test Accuracy of SVC =  0.9685
    -------------------------------
    Color space:  HSV
    Test Accuracy of SVC =  0.9854
    -------------------------------
    Color space:  HLS
    Test Accuracy of SVC =  0.9842
    -------------------------------
    Color space:  YCrCb
    Test Accuracy of SVC =  0.9899
    -------------------------------
    Color space:  YUV
    Test Accuracy of SVC =  0.9845
    -------------------------------
    Color space:  LUV
    Test Accuracy of SVC =  0.9837
    -------------------------------

As an aside, I also went through each of the individual color channels for HOG extraction but none of them had a test accuracy above 95%. This makes sense as `ALL` channels encompasses three times more information than any of the single-color channels.

Finally I explored six orientations (`orientations=[7, 8, 9, 10, 11, 12]` ) and fixing the other HOG parameters to `color_space = YCrCb`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. This is in cell 8 of the notebook. I got the following test accuracy results with highest test accuracy for 10 orientations. It is worth noting that picking 9 orientations is not a bad choice as it did get the highest test accuracy sometimes.  

    Orientation 7
    Test Accuracy of SVC =  0.9859
    ------------------------------
    Orientation 8
    Test Accuracy of SVC =  0.9848
    ------------------------------
    Orientation 9
    Test Accuracy of SVC =  0.987
    ------------------------------
    Orientation 10
    Test Accuracy of SVC =  0.9904
    ------------------------------
    Orientation 11
    Test Accuracy of SVC =  0.9873
    ------------------------------
    Orientation 12
    Test Accuracy of SVC =  0.9901
    ------------------------------

As you notice the test accuracy has progressively gotten better!

#### 3. Spatial Binning and Color Histograms.

I add spatial binning and color histogram features to the HOG features to see if the test prediction accuracy improves further. See cell 9 in the notebook for this code.

The test accuracy improved to 99.44% with spatial binning and histogram of color transforms. My final parameter selection for the classifier is as follows: 
* color_space = YCrCb
* orientations=10
* pixels_per_cell=(8, 8)
* cells_per_block=(2, 2)
* spatial bins = (32,32)
* histogram bins = 32

The classifier with the above parameters will be used to predict cars in the video.

#### 4. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is shown in cell 9. I combine features from HOG, spatial binning, and histogram of colors and normalize them. I split the normalized data into training and test sets with 80% of data for training and the rest for testing. With the training data I run the classifier. With the test data I find the accuracy of this classifier to be 99.44%

Normalizing the combined feature set is important to make sure it has a Gaussian distribution with zero mean and unit variance. Many machine learning algorithms do not perform well if the data is not centered at zero mean with unit variance. This is because the features with large variance dominate and have undue influence on the classifier. The function `StandardScaler()` from sklearn package is used to scale the data. Note that I use the same scaler obtained from the training data on the prediction data as well.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window function `find_cars` is implemented in cell 10. As described in the [lesson](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/c3e815c7-1794-4854-8842-5d7b96276642), I only extract hog features once and then sub-sample to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that is 8 x 8 cells. I step 2 cells per step which results in 75% (6/8) overlap between windows. A 75% overlap is granular enough to capture cars in the entire image. I use four scaling factors and the corresponding regions shown below:

* Scale 1.0 with ystart = 400 and ystop = 528
* Scale 1.5 with ystart = 400 and ystop = 592
* Scale 2.0 with ystart = 400 and ystop = 656
* Scale 2.5 with ystart = 336 and ystop = 656

The code for this is shown in cell 11. I start searching from about the middle (y=336) of the image in the y direction to the bottom of the image (y=656). I use the smallest scale (1.0) in the horizon of the image as the cars are small at that distance. I use progressively larger scales to find larger size cars which appear closer to the driver. 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To improve the performance of the classifier I fine-tuned the following parameters in the order shown below:
* pixels_per_cell=(8, 8)
* cells_per_block=(2, 2)
* color_space = YCrCb
* orientations=10
* spatial bins = (32,32) and histogram bins = 32

I started with a test accuracy of 98.42% and ended up with 99.44%. Note that there will be slight variations in these accuracy numbers every time you run the classifier.

The following picture shows the original image and its bounding boxes obtained by the classifier with the above parameters:

![alt text](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking/blob/master/output_images/CarsWithBB.jpg)

As seen from the above picture there are too many false positives (boxes around areas where there are no cars) and duplicate boxes (multiple boxes around cars). In the following I apply heatmaps to reduce/eliminate the false positives and duplicate boxes.

---

### Video Implementation

#### 1. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code is in cells 12 and 13. 

Here are some examples of the original image, its bounding boxes, heatmap and the final image with bounding boxes which is obtained by the result of `scipy.ndimage.measurements.label()`:

![alt text](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking/blob/master/output_images/hm2hm1hm3_s.jpg)

As you see the final bounding boxes are correct only in the picture 3. The other two pictures have three issues:
1. **Non-car false positives**: In picture 1 a non-car is identified as car. This is because the classifier is not accurate to identify the non-car
2. **Car false positives**: In picture 2 there is a bounding box to the left. This is because of  cars coming in the opposite direction. The classifier is accurately predicting them as cars but they need to be discarded
3. **Weak positives**: In picture 2 the black car is correctly identified but a very small box is drawn around it misjudging the size of the actual car

Through proper heatmap thresholding I fix issues 1 and 3. I collect bounding boxes for the last several frames and apply heat to the combined list of bounding boxes with a threshold of `35`. This eliminated the non-car false positives (issue 1) and enhanced correct predictions drawing the right size box (issue 3).

Fixing issue 2 is tricky as the classifier predicted correctly the cars in the opposite direction. However, I used the fact that the cars coming in the opposite direction are in the driver's field of view for a shorter time (fewer frames) than the cars in front of the driver. By adding bounding boxes for the past 20 frames I got rid of the false positives in the opposite direction. The reason being over 20 frames the heat generated by cars in the opposite direction was not strong enough compared to that of the cars in the same direction.

#### 2. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's the [link to my video result](https://www.youtube.com/embed/ew9HGIpdNQE?ecver=1). You'll notice the above issues were all fixed. The entire pipeline for a single image is implemented in cell 14 and this is called for every frame of the video in cell 20 of the notebook. To obtain the final video just run these cells in this order: 1, 2, 3, 9, 10, 12, 14, 18, 19, 20.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had to be careful with the image formats (.png vs .jpeg) and image conversions (`bgr` when using `cv2` and `rgb` when using `matplotlib`). 

Finding the optimum classifier while straightforward, was time consuming as I had to go through all HOG parameters one at a time to find the best option. 

Once I had a good classifier the false positives arising due to non-cars was easy to fix by tweaking the threshold. However, eliminating false positives arising from cars in the opposite direction took some time as I had to find the optimum number of frames to accumulate the bounding box. But I am afraid this is not robust enough. This may be fixed by constraining our search area in the `x` direction similar to what I did in the `y` direction.

