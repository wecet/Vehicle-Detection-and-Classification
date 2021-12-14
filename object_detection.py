# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import time

from imutils.video import VideoStream

# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

def object_detection(filename, conf_t=0.5, thresh=0.3, fr_limit=300):
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    configpath = os.path.join("YOLO Model", "darknet", "cfg", "yolov3.cfg")
    weightspath = os.path.join("YOLO Model", "darknet", "yolov3.weights")
    

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(configpath, weightspath)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output/objectdetection.avi', fourcc, 30, (800, 600), True)

    # image = cv2.imread(args["image"])
    vid = cv2.VideoCapture(filename)
    ret = True
    
    fr_no = 0
    
    start = time.time()

    while(ret):
        ret, frame = vid.read()
        if not ret:
            print("Error")
                
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
            (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > conf_t:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                #print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
        writer.write(cv2.resize(frame,(800, 600)))
        
        if fr_no >= fr_limit:
            break
        
        fr_no += 1
        
        # show the output image
        #cv2.imshow("Output", frame)
        #cv2.waitKey(0)
        
    
    print("[INFO] YOLO took {:.6f} seconds".format(time.time() - start))
    
    writer.release()
    vid.release()
    

viddirpath = os.path.join(os.path.dirname(os.getcwd()), "Data")
vidname = "20200323_155250.mp4"

object_detection(os.path.join(viddirpath, vidname), fr_limit=500)