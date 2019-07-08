# Incipient_slip_detection_with_GelSlim

This is the code for the paper "Maintaining Grasps within Slipping Bound by Monitoring Incipient Slip" in 2019 ICRA. 

Paper link: https://arxiv.org/pdf/1810.13381.pdf


## System Requirements
We use python 2.7 and ROS Kinetic on Ubuntu 18.04

software: OpenCV >= 3.0, matplotlib


## Sensor calibration
1. capture a GelSlim image with no contact (I used 427*320 resolution). Save it with the name of "reference_image.png" or other names. Change the name in line 32 in the "sensor_calibration.py" file.

2. run sensor_calibration.py and it will let you click four points in the image. The four points should be the four corners of a square shape (in physical world). I usualy select 4 black markers since I know the gaps bwteen markers are all the same. 
