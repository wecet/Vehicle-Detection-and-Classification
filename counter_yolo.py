# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import time

from imutils.video import VideoStream

class_dict = {'car': 0,
              'bicycle': 1,
              'motorbike': 2,
              'bus': 3,
              'truck': 4}

# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/


def object_detection(filename, conf_t=0.5, thresh=0.3, fr_limit=300):
    labelspath = "YOLO Model/darknet/coco.names"
    configpath = "YOLO Model/darknet/yolov3-320.cfg"
    weightspath = "YOLO Model/darknet/yolov3-320.weights"

    LABELS = open(labelspath).read().strip().split("\n")

    COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output/sunny_objectdetection_yolo.avi', fourcc, 30, (800, 600), True)

    vid = cv2.VideoCapture(filename)
    ret = True

    fr_no = 0

    start = time.time()

    while (ret):
        ret, frame = vid.read()
        if not ret:
            print("Error")

        (H, W) = frame.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > conf_t:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])

                    confidences.append(float(confidence))
                    classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_t, thresh)

        LS = []
        # ensure at least one detection exists
        if len(idxs) > 0:
            counter = len(idxs)
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                if LABELS[classIDs[i]] in class_dict.keys():
                    L = [str(class_dict[LABELS[classIDs[i]]]), str(confidences[i]), str(x / W), str(y / H), str(w / W), str(h / H)]
                    LS.append(" ".join(L))
                
                cv2.putText(frame, 'Vehicles Detected: ' + str(counter), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(255, 0, 0), thickness=2)

        writer.write(cv2.resize(frame, (800, 600)))
        
        output_txt = open("Predicted_Anns_YOLO/footage2/" + str(fr_no+1).zfill(4) + ".txt", "w")
        if len(LS) > 0:
            output_txt.write("\n".join(LS))
        else:
            output_txt.write("")
        output_txt.close()

        if fr_no >= fr_limit:
            break

        fr_no += 1

    print("[INFO] YOLO took {:.3f} minutes".format((time.time() - start) / 60))

    writer.release()
    vid.release()

viddirpath = os.path.join(os.path.dirname(os.getcwd()), "Videos")
vidname = "footage2.mp4"

object_detection(os.path.join(viddirpath, vidname), fr_limit=100)