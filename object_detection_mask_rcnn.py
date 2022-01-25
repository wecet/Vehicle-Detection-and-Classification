import os
import time
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

def object_detection(filename, conf_t=0.5, thresh=0.3, fr_limit=500):
    
    labelspath = os.path.join("Mask RCNN", "mask_rcnn", "mask-rcnn-coco","object_detection_classes_coco.txt")
    colorspath = os.path.join("Mask RCNN", "mask_rcnn", "mask-rcnn-coco","colors.txt")
    weightspath = os.path.join("Mask RCNN", "mask_rcnn", "mask-rcnn-coco","frozen_inference_graph.pb")
    configpath = os.path.join("Mask RCNN", "mask_rcnn", "mask-rcnn-coco","mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
    
    labels = open(labelspath).read().strip().split("\n")
    
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromTensorflow(weightspath, configpath)
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output/objectdetection_mask.avi', fourcc, 30, (800, 600), True)

    vid = cv2.VideoCapture(filename)
    ret = True
    
    fr_no = 0
    
    start = time.time()

    while(ret):
        ret, frame = vid.read()
        if not ret:
            print("Error")

        (imH, imW) = frame.shape[:2]

        # construct a blob from the input image and then perform a forward
        # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
        # of the objects in the image along with (2) the pixel-wise segmentation
        # for each specific object
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=True)
        net.setInput(blob)
        (boxes, masks) = net.forward(['detection_out_final', 'detection_masks'])

        # loop over the number of detected objects
        for i in range(boxes.shape[2]):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]
        
            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence <= conf_t:
                continue

            # clone our original image so we can draw on it
            frm_clone = frame.copy()
        
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([imW, imH, imW, imH])
            (startX, startY, endX, endY) = box.astype("int")
            boxW, boxH = endX - startX, endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > thresh)

            # extract the ROI of the image
            roi = frm_clone[startY:endY, startX:endX]

            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]
            
            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = colors[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
            frm_clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
            color = [int(c) for c in color]
            cv2.rectangle(frm_clone, (startX, startY), (endX, endY), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(labels[classID], confidence)
            cv2.putText(frm_clone, text, (startX, startY - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        writer.write(cv2.resize(frm_clone,(800, 600)))
    
        if fr_no >= fr_limit:
            break
        
        fr_no += 1
        
    print("[INFO] YOLO took {:.3f} minutes".format((time.time() - start)/60))
    
    writer.release()
    vid.release()
    

viddirpath = os.path.join(os.path.dirname(os.getcwd()), "Data")
vidname = "20200323_155250.mp4"

object_detection(os.path.join(viddirpath, vidname), fr_limit=500)