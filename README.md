# SelfDrivingCar-VehicleDetectionAndTracking

This is the fifth project in the Self Driving Car Nanodegree course offered by Udacity.

In this project vehicles are detected and tracked using computer vision and machine learning classification techniques. Here image features such as color and shape are extracted using Color Histogram, Spatial Binning and [Histogram of Oriented Gradients (HOG)](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) methods. A linear support vector machine (SVM) classifier is trained with these features to predict cars in the video stream.

You can check out the details on my implementation [here](https://github.com/kharikri/SelfDrivingCar-VehicleDetectionAndTracking/blob/master/writeup_report.md).

I implemented this project in Python, OpenCV, scikit-learn and FFmpeg.

Anyone interested in implementing this code requires the vehicle training data which can be obtained at the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).
