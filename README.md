# Vehicle Detection and Classification 

Submitted for fulfilment of the ARI3129 unit @ UoM

## Task 1 -> Getting the Data

The data used for this project is was data provided by the tutor. The data consisted of videos in different weather conditions and time settings which captured the CCTV footage of the Gozo Ferry docking area and the Msida junction 

## Task 2 -> Background Detection 

We had to start off by extracting the background from the videos. This was done by recognizing the average background frame thoughout the video and considering that as our background for the video. The code for this task can be found in background_detection.py

## Task 3 -> Object Detection 

The main functionality of this project is to detect vehicles. Many object detection frameworks are published but we decided to implement YOLOv3 by training it from scratch. To measure the accuracy we had to match the detected annotations with our manual annotations. We annotated the data using the EVA framework. 
https://github.com/Ericsson/eva

## Task 4 -> Compare Object Detection Techniques 

Besides implementing YOLOv3, in order to properly compare the frameworks' performance over other frameworks, we had to implement some more. We decided to implement YOLOv4 and FasterRCNN as our deep learning models alongside the HoG as our more traditional approach. The accuracy and performance of the models was measured using the mAP metric. An example of the mAP can be seen in the image below (in rainy conditions - YOLOv4 model)
![rain_YOLO](https://user-images.githubusercontent.com/73174341/159901467-4ad3795f-3c16-418b-abdc-4c762ccab282.png)

## Task 5 -> Video Analysis 

The last step to this project we had to count how many cars are in a frame. Tackling this we simply counted the number of boxes being drawn by the model and annotating it to the top right of the video output. Additionally, we had to also detect and differentiate how many motorcycles and cars are being detected in the frame. For this we updated the video annotation by appending the number of bounding boxes written for a certain class of vehicle.

All of the code can be found in the notebook combining all of the tasks. assignment.ipynb


