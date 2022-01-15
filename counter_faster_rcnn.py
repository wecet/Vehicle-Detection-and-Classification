# import necessary libraries
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import os
import time
import warnings

warnings.filterwarnings("ignore")

# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
# Note that there are some N/A's
# for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# we will use the same list for this notebook
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_cv, threshold):
    """
        get_prediction
        parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
        are chosen.
    """
    transform = T.Compose([T.ToTensor()])
    img = transform(img_cv)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    print("iteration")
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def object_detection_api(vid_path, threshold=0.7, rect_th=2, text_size=0.5, text_th=2, fr_limit=500):
    """
    object_detection_api
    parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
    method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written
        with opencv
        - the final image is displayed
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('output/objectdetection_faster_rcnn.avi', fourcc, 30, (800, 600), True)

    # image = cv2.imread(args["image"])
    vid = cv2.VideoCapture(vid_path)
    ret = True

    fr_no = 0

    start = time.time()
    print("[INFO] Started video processing...")

    while (ret):

        ret, frame = vid.read()
        if not ret:
            print("Error")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, pred_cls = get_prediction(Image.fromarray(frame), threshold)
        color_index = set(pred_cls)
        COLORS = np.random.uniform(0, 255, size=(len(color_index), 3))
        counter = len(boxes)

        for i in range(len(boxes)):
            pt1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
            pt2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
            cv2.rectangle(frame, pt1, pt2, color=list(color_index).index(pred_cls[i]), thickness=rect_th)
            cv2.putText(frame, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                        list(color_index).index(pred_cls[i]), thickness=text_th)

        cv2.putText(frame, 'Vehicles Detected: '+ str(counter), (50,50), cv2.FONT_HERSHEY_COMPLEX, text_size, color=(255,0,0), thickness=text_th)
        writer.write(cv2.resize(frame, (800, 600)))

        if fr_no % 100 == 0:
            print(fr_no)

        if fr_no >= fr_limit:
            break

        fr_no += 1

    print("[INFO] FasterRCNN took {:.3f} minutes".format((time.time() - start) / 60))

    writer.release()
    vid.release()


object_detection_api("E:/vehicle-detection-classification-opencv/CV Vids/20200323_155250.mp4")
